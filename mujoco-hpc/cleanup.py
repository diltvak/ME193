"""Free space on your cluster account: shows your quota, deletes old job results and caches, shows your quota again.

    python3 cleanup.py            # shows what it will delete and asks first
    python3 cleanup.py --yes      # no question

It deletes, on the cluster:
  - <HPC_DIR>/runs/        every finished job's videos, logs and checkpoints (copy anything you want first;
                           runIt.py already copied each run to ./runs/ on your laptop)
  - <HPC_DIR>/*.out        the Slurm output files of the test jobs
  - ~/.cache               pip and other download caches (anything needed is rebuilt, a little slower once)
It never touches ~/mujoco-env (your Python environment), ~/.ssh, or anything else.

Run it from this folder on your laptop (off campus: connect the VPN first).
"""
import argparse
import sys

from hpc import Cluster, StepFailed

QUOTA = "quota -s 2>/dev/null | tail -n +3 || true"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="do not ask before deleting")
    args = ap.parse_args()

    c = Cluster("cleanup")
    print(f"log file: {c.logpath}\n")
    targets = f"{c.dir}/runs {c.dir}/*.out ~/.cache"
    ok = False
    try:
        c.connect()
        print("Quota before:")
        c.ssh(QUOTA, check=False)

        print("\nThese would be deleted (size, path):")
        c.ssh(f"du -sh {targets} 2>/dev/null | sort -h; ls {c.dir}/runs 2>/dev/null | wc -l | "
              "sed 's/^/number of runs: /'", check=False)

        if not args.yes and input("\nDelete them? [y/N] ").strip().lower() != "y":
            c.say("nothing deleted")
            return
        c.ssh(f"rm -rf {c.dir}/runs {c.dir}/*.out ~/.cache")

        print("\nQuota after (can take a minute to update on the cluster's file system):")
        c.ssh(QUOTA, check=False)
        ok = True
    except StepFailed as e:
        c.say(f"FAILED: {e}")
    except KeyboardInterrupt:
        c.say("interrupted")
    print("\n" + ("Done." if ok else "Not finished; see the output above."))
    print(f"Full log: {c.logpath}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
