import json, sys, numpy as np, pandas as pd
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(f"{SRC}/raw_results.json"))
R = d["results"]; seeds = list(R)
methods = list(R[seeds[0]])
metrics = ["utility_auc", "mia_auc", "reid", "attr_disc", "linkability", "text_dx_adv", "text_loc_adv", "text_util"]
rows = []
for m in methods:
    row = {"method": m}
    for k in metrics:
        v = np.array([R[s][m][k] for s in seeds], float)
        row[k] = v.mean(); row[k + "_sd"] = v.std()
    row["reversible"] = R[seeds[0]][m]["reversible"]
    row["released"] = np.mean([R[s][m]["released"] for s in seeds])
    rows.append(row)
T = pd.DataFrame(rows).set_index("method")

# Encoding Privacy Gap Index (EPGI)
raw_auc = T.loc["Raw (no encoding)", "utility_auc"]; raw_tu = T.loc["Raw (no encoding)", "text_util"]
T["mia_adv"] = (2 * (T.mia_auc - 0.5)).clip(0, 1)
risk_cols = ["reid", "attr_disc", "mia_adv", "linkability", "text_dx_adv", "text_loc_adv"]
T["residual_risk"] = T[risk_cols].mean(axis=1)
T["P"] = 1 - T.residual_risk
T["U_struct"] = ((T.utility_auc - 0.5) / (raw_auc - 0.5)).clip(0, 1)
T["U_text"] = ((T.text_util - 0.5) / (raw_tu - 0.5)).clip(0, 1)
T["U"] = (T.U_struct + T.U_text) / 2
T["balance"] = 2 * T.P * T.U / (T.P + T.U)
T["EPGI"] = 1 - T.balance
T["max_threat"] = T[risk_cols].max(axis=1)
T.to_csv(f"{SRC}/summary.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(T[["utility_auc", "utility_auc_sd", "mia_auc", "reid", "attr_disc", "linkability", "text_dx_adv", "text_loc_adv",
         "text_util", "P", "U", "EPGI", "max_threat", "released"]].round(3))
print("CALE scrub:", {s: d["extra"][s].get("CALE (proposed)|scrub") for s in seeds})
print("CALE info:", {s: d["extra"][s].get("CALE (proposed)|scrub_info") for s in seeds})
print("examples:")
for m in methods:
    print(m, "::", R[seeds[0]][m]["example"])

short = {"Raw (no encoding)": "Raw", "Redaction": "Redaction", "Partial masking": "Masking",
         "Tokenization (vault)": "Tokenization", "Keyed hashing (HMAC)": "HMAC", "FPE (FF3-1)": "FPE",
         "Generalization (k=5)": "Gen. k=5", "Local DP (eps=1)": "LDP eps=1", "Local DP (eps=4)": "LDP eps=4",
         "Synthetic data (BN)": "Synthetic", "CALE (proposed)": "CALE", "CALE-S (proposed)": "CALE-S"}
plt.rcParams.update({"font.family": "serif", "font.size": 8})

# Fig 1: privacy-utility frontier
fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)
cluster = ["Raw (no encoding)", "Partial masking", "Tokenization (vault)", "Keyed hashing (HMAC)", "FPE (FF3-1)", "Redaction"]
lab_pos = {"Raw (no encoding)": (0.50, 0.04), "Tokenization (vault)": (0.50, 0.11), "Keyed hashing (HMAC)": (0.50, 0.18),
           "Partial masking": (0.50, 0.25), "FPE (FF3-1)": (0.50, 0.32), "Redaction": (0.50, 0.39),
           "Generalization (k=5)": (0.86, 0.60), "Local DP (eps=1)": (0.40, 0.74), "Local DP (eps=4)": (0.93, 0.74),
           "Synthetic data (BN)": (0.60, 0.86), "CALE (proposed)": (0.56, 0.95), "CALE-S (proposed)": (0.93, 0.92)}
for m in methods:
    prop = "CALE" in m
    x, y = T.loc[m, "U"], T.loc[m, "P"]
    ax.scatter(x, y, s=46 if prop else 20, marker="*" if prop else "o",
               color="#b2182b" if prop else "#2b5c8a", edgecolor="black", linewidth=0.4, zorder=3)
    tx, ty = lab_pos[m]
    ax.annotate(short[m], xy=(x, y), xytext=(tx, ty), fontsize=6.2,
                arrowprops=dict(arrowstyle="-", linewidth=0.35, color="0.45", shrinkA=0, shrinkB=2))
ax.set_xlabel("Utility retention U (structured + text)")
ax.set_ylabel("Privacy protection P")
ax.set_xlim(0.35, 1.05); ax.set_ylim(0, 1.06)
ax.grid(alpha=0.3, linewidth=0.4)
fig.tight_layout(); fig.savefig(f"{SRC}/fig_frontier.png"); plt.close(fig)

# Fig 2: residual-risk heatmap
labels = ["Re-ID", "Attr. discl.", "MIA adv.", "Linkability", "Text: dx", "Text: ZIP"]
H = T[risk_cols].values
fig, ax = plt.subplots(figsize=(3.45, 3.0), dpi=300)
im = ax.imshow(H, cmap="Greys", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
ax.set_yticks(range(len(methods))); ax.set_yticklabels([short[m] for m in methods], fontsize=6.5)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                color="white" if H[i, j] > 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(f"{SRC}/fig_heatmap.png"); plt.close(fig)

# Fig 3: CALE convergence
fig, ax = plt.subplots(figsize=(3.45, 1.7), dpi=300)
for s in seeds:
    h = (d["extra"][s].get("CALE (proposed)|scrub_info") or {}).get("hist", [])
    ax.plot(range(len(h)), h, marker="o", markersize=2.5, linewidth=0.9, label=f"seed {s}")
ax.axhline(0.05, color="#b2182b", linestyle="--", linewidth=0.8)
ax.text(0.2, 0.08, "stopping threshold delta = 0.05", fontsize=6, color="#b2182b")
ax.set_xlabel("Scrubbing iteration"); ax.set_ylabel("Internal adversary adv.")
ax.legend(fontsize=6, frameon=False); ax.grid(alpha=0.3, linewidth=0.4)
fig.tight_layout(); fig.savefig(f"{SRC}/fig_cale.png"); plt.close(fig)
print("figs ok")


# ---------------- v2 additions: timing, scrub recovery, EPGI weighting
tk = ["t_struct", "t_text", "t_scrub", "t_eval"]
TT = pd.DataFrame({m: {k: np.mean([R[s][m][k] for s in seeds]) for k in tk} for m in methods}).T
TT.to_csv(f"{SRC}/timing.csv"); print(TT.round(2))
meds = "metformin glipizide lisinopril amlodipine albuterol fluticasone sertraline fluoxetine meloxicam naproxen atorvastatin rosuvastatin tiotropium salmeterol sevelamer calcitriol".split()
land = "riverside harbor cypress pinecrest lakeview oakmont seagrape palm grove banyan coral bay heron point mangrove park marina library mall stadium".split()
rec = {}
for s in seeds:
    sc = d["extra"][s]["CALE (proposed)|scrub"]; inf = d["extra"][s]["CALE (proposed)|scrub_info"]
    rec[s] = dict(n=len(sc), prec=sum(t in meds + land for t in sc) / len(sc), med=sum(t in sc for t in meds) / 16,
                  land=sum(t in sc for t in land) / 20, rounds=len(inf["hist"]), loop_s=sum(inf["round_s"]),
                  apply_s=inf["apply_s"], hist=inf["hist"], other=[t for t in sc if t not in meds + land])
print(rec)
json.dump(rec, open(f"{SRC}/scrub_recovery.json", "w"), indent=1)
WS = {"Equal": [1, 1, 1, 1, 1, 1], "Structured x2": [2, 2, 2, 2, 1, 1], "Text x2": [1, 1, 1, 1, 2, 2],
      "Linkage-focused": [3, 2, 1, 2, 1, 1], "Membership x3": [1, 1, 3, 1, 1, 1]}
def epgi_w(w):
    w = np.array(w, float) / np.sum(w); Rr = (T[risk_cols].values * w).sum(1); Pp = 1 - Rr; Uu = T.U.values
    return pd.Series(1 - 2 * Pp * Uu / (Pp + Uu), index=T.index)
EW = pd.DataFrame({k: epgi_w(v) for k, v in WS.items()}); EW.to_csv(f"{SRC}/epgi_weights.csv"); print(EW.round(3))
rng = np.random.default_rng(0); first = 0; beat = 0; base_names = [m for m in methods if "CALE" not in m]
for _ in range(10000):
    e = epgi_w(rng.dirichlet(np.ones(6)))
    first += e.idxmin() == "CALE-S (proposed)"; beat += e["CALE-S (proposed)"] < e[base_names].min()
json.dump(dict(first=first / 10000, beats_baselines=beat / 10000), open(f"{SRC}/epgi_dirichlet.json", "w")); print(first / 10000, beat / 10000)
