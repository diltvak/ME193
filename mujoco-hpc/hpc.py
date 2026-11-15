"""Shared helpers for setup.py and runIt.py: one SSH connection, rsync, Slurm, and a log file.

Standard library only. It shells out to your own `ssh` and `rsync`, so password / Duo
prompts work exactly as they do in a terminal. Every command and its output is written to
logs/<name>-<timestamp>.log, so when something fails you can send that one file.
"""
import datetime
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".sh", ".py", ".sbatch", ".txt", ".md"}
DONE_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}


class StepFailed(Exception):
    pass


WINDOWS_HELP = """These scripts need ssh connection sharing and rsync, which native Windows does not have.
Use WSL (Windows Subsystem for Linux), which takes about 10 minutes once:
  1. In PowerShell (as administrator):  wsl --install      then restart and finish the Ubuntu setup
  2. Open the Ubuntu app and run:       sudo apt update && sudo apt install -y rsync openssh-client python3
  3. Keep this folder INSIDE Ubuntu (for example ~/mujoco-hpc), not under /mnt/c, and run the same
     commands there: python3 setup.py   (VSCode: install the "WSL" extension, then "Connect to WSL").
"""


def open_file(path):
    """Open a video/image with the system viewer; if that is not possible just say where it is."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif "microsoft" in platform.release().lower():  # WSL
            win = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True).stdout.strip()
            subprocess.run(["explorer.exe", win], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False, capture_output=True)
    except OSError:
        print(f"  (open it yourself: {path})")


def load_config():
    """Read HPC_USER / HPC_HOST / HPC_DIR from config.sh (one place to edit). Env vars override."""
    cfg = {}
    for line in (HERE / "config.sh").read_text().splitlines():
        m = re.match(r'\s*export\s+(\w+)="([^"]*)"', line)
        if m:
            cfg[m.group(1)] = m.group(2)
    for key in ("HPC_USER", "HPC_HOST", "HPC_DIR"):
        cfg[key] = os.environ.get(key, cfg.get(key, ""))
        if not cfg[key]:
            raise SystemExit(f"{key} is not set in config.sh")
    if cfg["HPC_USER"].startswith("YOUR_"):
        raise SystemExit("Edit config.sh first: put your Tufts username in HPC_USER.")
    return cfg


class Cluster:
    def __init__(self, logname):
        if sys.platform == "win32":
            raise SystemExit(WINDOWS_HELP)
        if shutil.which("sbatch"):
            raise SystemExit(f"STOP: this script is for your laptop, but you are on the cluster ({platform.node()}).")
        cfg = load_config()
        self.user, self.host, self.dir = cfg["HPC_USER"], cfg["HPC_HOST"], cfg["HPC_DIR"].rstrip("/")
        self.target = f"{self.user}@{self.host}"
        self.sock = os.path.expanduser("~/.ssh/sockets/%r@%h")
        os.makedirs(os.path.dirname(self.sock), exist_ok=True)
        (HERE / "logs").mkdir(exist_ok=True)
        self.logpath = HERE / "logs" / f"{logname}-{datetime.datetime.now():%Y%m%d-%H%M%S}.log"
        self.logf = open(self.logpath, "w")
        self.log(f"{logname} on {platform.node()}, python {sys.version.split()[0]}, target {self.target}, dir {self.dir}")

    # ---- output -------------------------------------------------------------------------
    def log(self, text):
        self.logf.write(text + "\n")
        self.logf.flush()

    def say(self, text):
        print(text, flush=True)
        self.log(text)

    # ---- running commands ---------------------------------------------------------------
    def run(self, argv, check=True, show=True):
        """Run a local command, streaming and logging its output. Returns (returncode, output)."""
        self.log("\n$ " + " ".join(shlex.quote(a) for a in argv))
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        for line in p.stdout:
            lines.append(line)
            self.logf.write(line)
            self.logf.flush()
            if show:
                print("    " + line, end="", flush=True)
        rc = p.wait()
        self.log(f"[exit {rc}]")
        if check and rc != 0:
            raise StepFailed(f"command failed (exit {rc}): {' '.join(argv[:3])} ...")
        return rc, "".join(lines)

    def _ssh_base(self):
        return ["ssh", "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", f"ControlPath={self.sock}"]

    def ssh(self, cmd, **kw):
        """Run a shell command on the cluster (login shell, so `module` works)."""
        return self.run(self._ssh_base() + [self.target, "bash -lc " + shlex.quote(cmd)], **kw)

    def rsync(self, src, dst, extra=()):
        ssh = " ".join(shlex.quote(a) for a in self._ssh_base())
        return self.run(["rsync", "-az", *extra, "-e", ssh, *([src] if isinstance(src, str) else src), dst], show=False)

    # ---- connection ---------------------------------------------------------------------
    def connected(self):
        r = subprocess.run(["ssh", "-O", "check", "-o", f"ControlPath={self.sock}", self.target], capture_output=True)
        return r.returncode == 0

    def connect(self):
        """Open one authenticated connection (2 hours) that every later command reuses."""
        if self.connected():
            self.say("reusing the SSH connection that is already open")
            return
        self.say(f"Opening SSH connection to {self.target}. Answer any password / Duo prompt here:")
        self.log("$ ssh -fN (interactive)")
        r = subprocess.run(["ssh", "-o", "ControlMaster=yes", "-o", "ControlPersist=2h",
                            "-o", f"ControlPath={self.sock}", "-fN", self.target])
        if r.returncode != 0 or not self.connected():
            raise StepFailed("could not log in. Off campus? Connect the Tufts VPN. Check HPC_USER / HPC_HOST in config.sh.")
        self.say("connected")

    # ---- files --------------------------------------------------------------------------
    def push(self, local_paths, remote_dir):
        self.ssh(f"mkdir -p {remote_dir}", show=False)
        # The cluster is Linux: a script with Windows (CRLF) line endings fails there with
        # "bad interpreter" / "$'\\r': command not found", so send text files with LF endings.
        with tempfile.TemporaryDirectory() as tmp:
            staged = []
            for p in map(Path, local_paths):
                if p.suffix in TEXT_SUFFIXES:
                    q = Path(tmp) / p.name
                    q.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
                    p = q
                staged.append(str(p))
            self.rsync(staged, f"{self.target}:{remote_dir}/")

    def pull(self, remote_dir, local_dir):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        self.rsync(f"{self.target}:{remote_dir}/", f"{local_dir}/")

    # ---- slurm --------------------------------------------------------------------------
    def submit(self, remote_dir, sbatch_file, *options):
        _, out = self.ssh(f"cd {remote_dir} && sbatch --parsable {' '.join(options)} {sbatch_file}", show=False)
        jobid = out.strip().splitlines()[-1].split(";")[0]
        if not jobid.isdigit():
            raise StepFailed(f"sbatch did not return a job id. It said: {out.strip()}")
        self.say(f"submitted job {jobid}")
        return jobid

    def wait(self, jobid, poll=10):
        """Block until the job leaves the queue. Returns {'state', 'exit', 'elapsed', 'node'}."""
        last = None
        try:
            while True:
                _, out = self.ssh(f'squeue -h -j {jobid} -o "%T|%R"', check=False, show=False)
                line = out.strip().splitlines()[-1] if out.strip() else ""
                if "|" not in line:
                    break  # no longer in the queue
                if line != last:
                    state, reason = line.split("|", 1)
                    self.say(f"  job {jobid}: {state} ({reason})")
                    last = line
                time.sleep(poll)
            for _ in range(10):  # accounting can lag a few seconds behind the queue
                _, out = self.ssh(f"sacct -j {jobid} -X -n -P -o State,ExitCode,Elapsed,NodeList", check=False, show=False)
                row = out.strip().splitlines()[-1].split("|") if out.strip() else []
                if len(row) == 4 and row[0].split()[0] in DONE_STATES:
                    break
                time.sleep(3)
            else:
                row = ["UNKNOWN", "?", "?", "?"]
        except KeyboardInterrupt:
            self.say(f"\nStopped watching. Job {jobid} is still on the cluster (cancel it with: scancel {jobid}).\nCheck on it any time with: python3 status.py {jobid}")
            raise
        res = {"state": row[0].split()[0], "exit": row[1], "elapsed": row[2], "node": row[3]}
        self.say(f"  job {jobid} finished: {res['state']} (exit {res['exit']}, {res['elapsed']}, node {res['node']})")
        return res

    def read_remote(self, path):
        _, out = self.ssh(f"cat {path}", check=False, show=False)
        self.log(f"--- {path} ---\n{out}\n--- end ---")
        return out
