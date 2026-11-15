#!/usr/bin/env python3
"""
Train a finger-count classifier (0-5) on dataset/<label>/*.jpg and export it
to ONNX for the Arduino Uno Q.

    python3 train.py                 # train, evaluate, export
    python3 train.py --epochs 40

Model: MobileNetV3-Small pretrained on ImageNet, fine-tuned end to end.
With only a few thousand images, transfer learning beats a CNN trained from
scratch by a wide margin, and this network is small enough (~1.5M params,
~20 MFLOPs at 128x128) to run in real time on the Uno Q's Cortex-A53 cores.

Outputs (in ./model):
    fingers.pt     best PyTorch weights
    fingers.onnx   portable model. Input:  float32 [N,3,128,128], RGB, 0..1
                                   Output: logits [N,6]
                   (ImageNet mean/std normalisation is baked into the graph.)

Validation split: frames are captured 10/s, so neighbours are near-identical
and a random split would leak. Each class's time-ordered frames are cut into
10 contiguous blocks and 2 whole blocks are held out for validation.
"""
import argparse
import os
from pathlib import Path

try:  # python.org builds on macOS lack root certs; needed for the weight download
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
NUM_CLASSES = 6
SIZE = 128
VAL_BLOCKS = (3, 8)  # of 10 contiguous time blocks per class


def load_split(root):
    """Return (train_x, train_y, val_x, val_y) as uint8 NCHW RGB tensors."""
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for label in range(NUM_CLASSES):
        files = sorted((root / str(label)).glob("*.jpg"))
        if not files:
            raise SystemExit(f"No images in {root / str(label)}")
        for i, f in enumerate(files):
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (SIZE, SIZE):
                img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            block = i * 10 // len(files)
            (va_x if block in VAL_BLOCKS else tr_x).append(img)
            (va_y if block in VAL_BLOCKS else tr_y).append(label)
    to_t = lambda xs: torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    return to_t(tr_x), torch.tensor(tr_y), to_t(va_x), torch.tensor(va_y)


class Frames(Dataset):
    def __init__(self, x, y, tf):
        self.x, self.y, self.tf = x, y, tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


class Deploy(nn.Module):
    """Wraps the network so the exported model takes plain RGB in 0..1."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build_model():
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, NUM_CLASSES)
    return Deploy(net)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p in zip(y, pred):
            conf[t, p] += 1
    return conf.diag().sum().item() / conf.sum().item(), conf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "dataset")
    ap.add_argument("--out", type=Path, default=HERE / "model")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tr_x, tr_y, va_x, va_y = load_split(args.data)
    print(f"device={device}  train={len(tr_y)}  val={len(va_y)}")

    # Heavy augmentation: the data comes from one background / lighting setup,
    # so vary geometry and colour to stop the net keying on those.
    train_tf = v2.Compose([
        v2.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.75, 1.15)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
        v2.RandomGrayscale(p=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ])
    val_tf = v2.ToDtype(torch.float32, scale=True)
    train_dl = DataLoader(Frames(tr_x, tr_y, train_tf), batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(Frames(va_x, va_y, val_tf), batch_size=256)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    args.out.mkdir(parents=True, exist_ok=True)
    pt_path, onnx_path = args.out / "fingers.pt", args.out / "fingers.onnx"
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)
        acc, _ = evaluate(model, val_dl, device)
        flag = ""
        if acc > best:
            best, flag = acc, "  *saved*"
            torch.save(model.state_dict(), pt_path)
        print(f"epoch {epoch:2d}/{args.epochs}  loss {total / n:.4f}  val acc {acc:.4f}{flag}")

    # Reload best weights, report, export.
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model = model.cpu().eval()
    acc, conf = evaluate(model, val_dl, "cpu")
    print(f"\nBest val accuracy: {acc:.4f}\nConfusion matrix (rows = true, cols = predicted):")
    print("     " + "".join(f"{c:5d}" for c in range(NUM_CLASSES)))
    for c in range(NUM_CLASSES):
        print(f"{c:5d}" + "".join(f"{v:5d}" for v in conf[c].tolist()))

    dummy = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model, (dummy,), str(onnx_path), input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
    )

    # Verify the exported file reproduces the PyTorch model on the validation set.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = va_x.float().div(255).numpy()
    onnx_logits = np.concatenate([sess.run(None, {"image": x[i:i + 256]})[0] for i in range(0, len(x), 256)])
    onnx_acc = (onnx_logits.argmax(1) == va_y.numpy()).mean()
    with torch.no_grad():
        diff = np.abs(model(torch.from_numpy(x[:64])).numpy() - onnx_logits[:64]).max()
    print(f"\nExported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"ONNX val accuracy: {onnx_acc:.4f}   max |torch - onnx| logit diff: {diff:.2e}")


if __name__ == "__main__":
    main()
