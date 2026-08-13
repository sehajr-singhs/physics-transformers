"""run_matrix.py — multi-seed training matrix for the baselines comparison.

Trains, for each domain x model kind x seed:
    phys    : PhysFormer (attention) + physics-consistency loss
    nophys  : PhysFormer (attention), physics term off
    mlp     : MLP encoder, data-only (no attention, no physics)

All runs use the same budget (256 samples, 60 epochs, d_model 48, 3 layers,
traj_hidden 64) so the only differences are attention and/or the physics
term. Results are parsed from the training logs into paper/fig/fig9_data.json
(mean +/- std over seeds), and the median-seed "phys" model for each domain is
promoted to physx/models/<domain>.pt so the headline figures use the same
configuration as the table.

usage: python physx/run_matrix.py [--seeds 0-4] [--workers 5]
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATRIX_DIR = os.path.join(ROOT, "physx", "models", "matrix")
LOG_DIR = os.path.join(ROOT, "physx", "models")

DOMAINS = ["beam", "cantilever", "projectile", "burgers", "heat2d"]
KINDS = ["phys", "nophys", "mlp"]
EPOCHS = 60
SAMPLES = 256
THREADS = 4


def job_tag(domain, kind, seed):
    return f"matrix_{domain}_{kind}_s{seed}"


def run_job(domain, kind, seed):
    tag = job_tag(domain, kind, seed)
    cmd = [
        sys.executable, os.path.join(HERE, "train.py"),
        "--domain", domain,
        "--epochs", str(EPOCHS),
        "--samples", str(SAMPLES),
        "--seed", str(seed),
        "--threads", str(THREADS),
        "--save", os.path.join(MATRIX_DIR, f"{tag}.pt"),
    ]
    if kind == "mlp":
        cmd += ["--kind", "mlp"]
    elif kind == "nophys":
        cmd += ["--w-phys", "0"]
    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    ok = r.returncode == 0
    # parse the last epoch line
    mae = phys = None
    if ok:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            last = None
            for line in f:
                m = re.search(r"val_rel_mae ([0-9.eE+-]+) phys_resid ([0-9.eE+-]+)", line)
                if m:
                    last = (float(m.group(1)), float(m.group(2)))
            if last:
                mae, phys = last
    print(f"[matrix] {tag}: {'OK' if ok else 'FAIL'} mae={mae} phys={phys} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return {"tag": tag, "domain": domain, "kind": kind, "seed": seed,
            "ok": ok, "val_rel_mae": mae, "phys_resid": phys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--domains", default=None, help="comma-separated subset of DOMAINS")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    domains = DOMAINS if not args.domains else [d.strip() for d in args.domains.split(",")]
    os.makedirs(MATRIX_DIR, exist_ok=True)

    jobs = [(d, k, s) for d in domains for k in KINDS for s in seeds]
    print(f"[matrix] {len(jobs)} jobs, {args.workers} workers", flush=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, d, k, s) for (d, k, s) in jobs]
        for f in cf.as_completed(futs):
            results.append(f.result())

    # aggregate: mean +/- std over seeds, per domain x kind
    agg = {}
    for d in DOMAINS:
        agg[d] = {}
        for k in KINDS:
            rows = [r for r in results if r["domain"] == d and r["kind"] == k and r["ok"]]
            maes = [r["val_rel_mae"] for r in rows if r["val_rel_mae"] is not None]
            phy = [r["phys_resid"] for r in rows if r["phys_resid"] is not None]
            def stats(xs):
                if not xs:
                    return None
                mu = sum(xs) / len(xs)
                sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
                return {"mean": mu, "std": sd, "values": xs, "n": len(xs)}
            agg[d][k] = {"val_rel_mae": stats(maes), "phys_resid": stats(phy)}

    # merge (never overwrite) so a subset run does not erase completed domains
    out = os.path.join(ROOT, "paper", "fig", "fig9_data.json")
    if os.path.exists(out):
        with open(out) as f:
            prev = json.load(f)
        prev.update(agg)
        agg = prev
    with open(out, "w") as f:
        json.dump(agg, f, indent=1)
    print(f"[matrix] wrote {out}", flush=True)

    # promote the median-seed phys model per domain to the headline location
    for d in DOMAINS:
        rows = sorted(
            [r for r in results if r["domain"] == d and r["kind"] == "phys" and r["ok"]
             and r["val_rel_mae"] is not None],
            key=lambda r: r["val_rel_mae"])
        if not rows:
            print(f"[matrix] no phys model for {d}, skipping promotion", flush=True)
            continue
        best = rows[len(rows) // 2]  # median by val error
        src_pt = os.path.join(MATRIX_DIR, f"{best['tag']}.pt")
        src_stats = os.path.splitext(src_pt)[0] + ".stats.json"
        dst_pt = os.path.join(LOG_DIR, f"{d}.pt")
        dst_stats = os.path.join(LOG_DIR, f"{d}.stats.json")
        if os.path.exists(src_pt) and os.path.exists(src_stats):
            import shutil
            shutil.copyfile(src_pt, dst_pt)
            shutil.copyfile(src_stats, dst_stats)
            print(f"[matrix] promoted {best['tag']} -> physx/models/{d}.pt "
                  f"(mae {best['val_rel_mae']:.4f})", flush=True)
    print("[matrix] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
