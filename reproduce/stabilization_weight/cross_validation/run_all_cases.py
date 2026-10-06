"""run_all_cases: run the run_cross_validation.py chunks at most THREE at a time (1 BLAS thread each).

No decision rule lives here; it only schedules.  Order: every part-A MN chunk, then every part-A PN
chunk, then every part-B CV chunk, so that if the run is stopped early part A is complete first.
Each solve inside a chunk is checkpointed, so re-running the driver resumes.

  python -B run_all_cases.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_cross_validation.py")
LOGS = os.path.join(HERE, "../../../outputs/stabilization_weight/cross_validation", "logs")
CASES = ["c1", "c2", "c3", "c4", "c5", "c6"]
MAXP = 3

JOBS = ([(c, "MN") for c in CASES] + [(c, "PN") for c in CASES] + [(c, "CV") for c in CASES])


def main():
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "1"
    running, queue, done = [], list(JOBS), []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < MAXP:
            case, kind = queue.pop(0)
            log = open(os.path.join(LOGS, "%s_%s.log" % (case, kind)), "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-B", RUN, "--case", case, "--kind", kind],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running.append((p, case, kind, log, time.time()))
            print("[%7.0fs] start %s %s pid %d  (queued %d)" % (time.time() - t0, case, kind, p.pid, len(queue)),
                  flush=True)
        time.sleep(5.0)
        for item in list(running):
            p, case, kind, log, ts = item
            if p.poll() is not None:
                running.remove(item)
                log.close()
                done.append((case, kind, p.returncode))
                print("[%7.0fs] done  %s %s rc %s  (%.0f s)  remaining %d+%d"
                      % (time.time() - t0, case, kind, p.returncode, time.time() - ts, len(queue), len(running)),
                      flush=True)
    print("ALL DONE in %.0f s" % (time.time() - t0), flush=True)
    for case, kind, rc in done:
        print("  %s %s rc %s" % (case, kind, rc), flush=True)
    r = subprocess.run([sys.executable, "-B", RUN, "--sum"], cwd=HERE, env=env,
                       capture_output=True, text=True)
    print(r.stdout[-8000:], flush=True)
    print(r.stderr[-4000:], flush=True)


if __name__ == "__main__":
    main()
