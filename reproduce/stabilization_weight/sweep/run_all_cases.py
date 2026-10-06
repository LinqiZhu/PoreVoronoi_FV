"""run_all_cases: run run_sweep.py for the given cases with at most CONC concurrent workers.

  python -B run_all_cases.py --cases c1,c2,c3,c4,c5,c6 --conc 3

(the default --cases is c4,c5,c6).  Not a reported-number script: it only schedules
run_sweep.py (whose sha256 is recorded) and writes
outputs/stabilization_weight/sweep/logs/<case>.log and .../logs/drive_status.json.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_sweep.py")
OUT = os.path.join(HERE, "../../../outputs/stabilization_weight/sweep")
LOGS = os.path.join(OUT, "logs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="c4,c5,c6")
    ap.add_argument("--conc", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    queue = [c for c in a.cases.split(",") if c]
    running, done = {}, {}
    while queue or running:
        while queue and len(running) < a.conc:
            c = queue.pop(0)
            lf = open(os.path.join(LOGS, "%s.log" % c), "a")
            p = subprocess.Popen([sys.executable, "-B", RUN, "--case", c], stdout=lf,
                                 stderr=subprocess.STDOUT, cwd=HERE, env=env, creationflags=flags)
            running[c] = (p, lf, time.time())
            print("[start]", c, p.pid, flush=True)
        for c in list(running):
            p, lf, t0 = running[c]
            rc = p.poll()
            if rc is not None:
                lf.close()
                done[c] = dict(rc=rc, wall_s=time.time() - t0)
                del running[c]
                print("[done]", c, rc, flush=True)
        with open(os.path.join(LOGS, "drive_status.json"), "w") as f:
            json.dump(dict(running=list(running), queued=queue, done=done,
                           t=time.strftime("%H:%M:%S")), f, indent=1)
        time.sleep(10)
    print("[all done]", json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
