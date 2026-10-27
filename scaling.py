"""Cost scaling of the CALE leakage-guided scrubbing loop (Reviewer 2 item 4, Reviewer 3).
Times the loop on controller calibration samples of increasing size and applies the learned
scrub list to a fixed 10,000-note release."""
import json, time, tracemalloc, sys
import numpy as np
import privenc_experiment as E

OUT = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
redact = lambda r: {"name": "[PERSON]", "age": "[AGE]", "zip": "[LOCATION]", "dx": "[CONDITION]",
                    "icd": "[CODE]", "acct": "[ACCOUNT]", "occ": r.occ}
rows = []
for seed in E.SEEDS:
    rng = np.random.default_rng(seed)
    pop = E.make_population(rng, 50000)
    rel = pop.iloc[:10000]
    rel_texts = np.array([E.render(r, redact(r)) for r in rel.itertuples()], dtype=object)
    for n in E.CFG["scaling"]["calibration_sizes"]:
        cal = pop.iloc[10000:10000 + n]
        t0 = time.perf_counter()
        cal_texts = np.array([E.render(r, redact(r)) for r in cal.itertuples()], dtype=object)
        t_render = time.perf_counter() - t0
        prot = [cal.dx.values.astype(str), cal.zip.values.astype(str)]
        t1 = time.perf_counter()   # timing pass (no memory tracing, which slows Python)
        _, scrub, info = E.leakage_guided_scrub(rel_texts, None, seed, cal_texts=cal_texts, cal_protected=prot)
        t_loop = time.perf_counter() - t1
        tracemalloc.start()        # separate memory pass
        E.leakage_guided_scrub(rel_texts, None, seed, cal_texts=cal_texts, cal_protected=prot)
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        rows.append(dict(seed=seed, n_cal=n, rounds=len(info["hist"]), loop_s=t_loop, round_s=info["round_s"],
                         apply_s=info["apply_s"], render_s=t_render, peak_mb=peak, scrub_len=len(scrub),
                         final_adv=info["hist"][-1]))
        print(rows[-1], flush=True)
json.dump(rows, open(f"{OUT}/scaling.json", "w"), indent=1)
print("done")
