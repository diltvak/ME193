"""Check on your cluster jobs at any time (from any laptop terminal, even after closing the one that started them).

    python3 status.py              # what is queued/running now, plus jobs from the last 2 days
    python3 status.py 6518997      # state of one job and the end of its output
    python3 status.py --watch      # refresh every 10 s until you press Ctrl-C
    python3 status.py --fetch 6518997   # copy that job's results to ./runs/ (if you stopped watching early)
    python3 status.py --cancel 6518997  # stop a job
"""
import argparse
import time
from pathlib import Path

from hpc import HERE, Cluster, StepFailed


def table(c, cmd, header):
    _, out = c.ssh(cmd, check=False, show=False)
    rows = [l.split("|") for l in out.strip().splitlines() if "|" in l]
    if not rows:
        return False
    rows.insert(0, header)
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for r in rows:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)))
    return True


def overview(c):
    print("\nIn the queue or running now:")
    if not table(c, 'squeue -h -u $USER -o "%i|%j|%P|%T|%M|%R"', ["JOBID", "NAME", "PARTITION", "STATE", "TIME", "REASON / NODE"]):
        print("  nothing. (Everything you submitted has finished.)")
    print("\nFinished in the last 2 days:")
    if not table(c, "sacct -u $USER -S now-2days -X -n -P -o JobID,JobName,Partition,State,Elapsed,NodeList",
                 ["JOBID", "NAME", "PARTITION", "STATE", "ELAPSED", "NODE"]):
        print("  none")
    print("\nTip: python3 status.py <JOBID> shows that job's output; PENDING + Resources/Priority just means waiting for a free node.")


def find_output(c, jobid):
    _, out = c.ssh(f"ls -t {c.dir}/*-{jobid}.out {c.dir}/runs/*/slurm-{jobid}.out 2>/dev/null | head -1", check=False, show=False)
    return out.strip()


def one_job(c, jobid, lines=25):
    print(f"\nJob {jobid}:")
    _, out = c.ssh(f"squeue -h -j {jobid} -o '%T (%r), running %M, on %N' 2>/dev/null", check=False, show=False)
    if out.strip():
        print("  now: " + out.strip())
    else:
        _, out = c.ssh(f"sacct -j {jobid} -X -n -P -o State,ExitCode,Elapsed,NodeList", check=False, show=False)
        row = out.strip().splitlines()[0].split("|") if out.strip() else None
        print(f"  finished: {row[0]} (exit {row[1]}, {row[2]}, node {row[3]})" if row else "  not found (job ids are only kept for a while)")
    path = find_output(c, jobid)
    if path:
        print(f"\nLast {lines} lines of {path}:")
        _, out = c.ssh(f"tail -n {lines} {path}", check=False, show=False)
        print("  " + (out.rstrip().replace("\n", "\n  ") or "(empty so far)"))
    else:
        print("  no output file yet (the job may still be waiting to start).")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", nargs="?", help="a job id, for details and its output")
    ap.add_argument("--watch", action="store_true", help="refresh every 10 seconds")
    ap.add_argument("--fetch", metavar="JOB", help="copy that job's run folder back to ./runs/")
    ap.add_argument("--cancel", metavar="JOB", help="cancel a job")
    a = ap.parse_args()

    c = Cluster("status")
    try:
        c.connect()
        if a.cancel:
            c.ssh(f"scancel {a.cancel}")
            c.say(f"cancel requested for job {a.cancel}")
        elif a.fetch:
            path = find_output(c, a.fetch)
            if "/runs/" not in path:
                raise StepFailed(f"no run folder found for job {a.fetch} (only runIt.py jobs have one)")
            rdir = str(Path(path).parent)
            local = HERE / "runs" / Path(rdir).name
            c.pull(rdir, local)
            c.say(f"copied to {local}")
            for p in sorted(local.iterdir()):
                c.say(f"  {p.name}")
        else:
            while True:
                if a.watch:
                    print("\033[2J\033[H", end="")
                    print(time.strftime("%H:%M:%S") + "  (Ctrl-C to stop)")
                one_job(c, a.job) if a.job else overview(c)
                if not a.watch:
                    break
                time.sleep(10)
    except StepFailed as e:
        c.say(f"FAILED: {e}")
    except KeyboardInterrupt:
        print()
    print(f"\nLog: {c.logpath}")


if __name__ == "__main__":
    main()
