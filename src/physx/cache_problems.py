"""cache_problems.py — generate the 10-law problem sets once per seed and
save them as JSON, so the six (real/dummy x 3 seeds) training runs reuse the
exact same data. Problems depend only on the seed, not on the law mode."""

import json
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physx import dataset, laws

DOMAINS = laws.SHARED_HEAD_DOMAINS
PER_DOMAIN = 96
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ext", "problems_s{seed}.json")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for seed in (0, 1, 2):
        path = OUT.format(seed=seed)
        if os.path.exists(path):
            print(f"exists: {path}", flush=True)
            continue
        t0 = time.time()
        probs = {d: dataset.generate(d, n=PER_DOMAIN, seed=seed + 100 * i)
                 for i, d in enumerate(DOMAINS)}
        with open(path, "w") as f:
            json.dump({"seed": seed, "per_domain": PER_DOMAIN, "domains": DOMAINS,
                       "problems": probs}, f)
        print(f"saved {path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
