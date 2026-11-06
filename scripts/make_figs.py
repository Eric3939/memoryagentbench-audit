"""Paper figures (vector PDF) from the zero-cost analyses. Run with the venv python.

fig_claims.pdf  claimed gain vs n with the 95% item-noise boundary
fig_repro.pdf   published vs independently reproduced scores (memory systems vs RAG/LC)
fig_ranks.pdf   overall-rank probability heatmap for MAB Table 3 (deff 1)
"""
import csv, json, math, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = f"{ROOT}/paper/figs"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})

# claims
d = json.load(open(f"{ROOT}/reports/data/claims_audit.json"))
C = d["claims"]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
n = np.geomspace(20, 4000, 200)
ax.plot(n, 100 * 1.96 * np.sqrt(0.5 / n), color="#333", lw=1, label="95% item-noise boundary (p=.5)")
for within, col, lab in [(True, ORANGE, "within item noise"), (False, BLUE, "clears item noise")]:
    pts = [c for c in C if (abs(c["z_u"]) < 1.96) == within]
    rng = np.random.default_rng(0)
    ax.scatter([c["n"] * math.exp(rng.normal(0, 0.06)) for c in pts], [min(c["gap"], 40) for c in pts], s=9,
               color=col, alpha=.85, lw=0.4, edgecolor="white", label=f"{lab} ({len(pts)})")
ax.set_xscale("log"); ax.set_xlim(20, 4000); ax.set_ylim(0, 41)
ax.set_xticks([25, 50, 100, 200, 500, 1000, 3000]); ax.set_xticklabels(["25", "50", "100", "200", "500", "1k", "3k"])
ax.set_xlabel("questions behind the score (n)"); ax.set_ylabel("claimed gain (points)")
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_claims.pdf"); plt.close(fig)

# reproductions
R = []
for r in csv.DictReader(open(f"{ROOT}/papers/search/reproductions.csv", encoding="utf-8")):
    try:
        a, b = float(r["published_score"]), float(r["reproduced_score"])
    except ValueError:
        continue
    if a <= 1 and b <= 1:
        a, b = a * 100, b * 100
    mem = not re.search(r"BM25|RAG|Full-context|MemPalace raw|flat retriever", r["system"])
    R.append((mem, b - a, a, b, r["system"].split(" (")[0]))
R.sort(key=lambda x: (x[0], -x[1]))  # RAG/LC rows at bottom, memory rows on top
fig, ax = plt.subplots(figsize=(3.3, 2.6))
for i, (mem, dlt, a, b, s) in enumerate(R):
    col = ORANGE if mem else BLUE
    ax.plot([a, b], [i, i], color=col, lw=1, alpha=.7)
    ax.scatter([a], [i], s=10, facecolor="white", edgecolor=col, lw=0.8, zorder=3)
    ax.scatter([b], [i], s=10, color=col, zorder=3)
ax.set_yticks([]); ax.set_xlim(0, 100); ax.set_xlabel("score (open = published, filled = reproduced)")
mm = np.median([abs(x[1]) for x in R if x[0]]); mr = np.median([abs(x[1]) for x in R if not x[0]])
nm = sum(x[0] for x in R)
ax.axhline(len(R) - nm - 0.5, color=GREY, lw=0.5, ls=":")
ax.set_title(f"LLM-built memory (n={nm}): median |Δ| {mm:.1f} pts", color=ORANGE, fontsize=7, loc="left", pad=11)
ax.text(0, 1.005, f"BM25 / RAG / long context (n={len(R)-nm}): median |Δ| {mr:.1f} pts", color=BLUE, fontsize=7,
        transform=ax.transAxes, va="bottom")
ax.set_ylim(-1, len(R))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_repro.pdf"); plt.close(fig)

# rank heatmap
na = json.load(open(f"{ROOT}/reports/data/noise_audit.json"))
ag = sorted(na["agents"], key=lambda a: na["reported_rank"][a])
P = np.array([na["scenarios"]["1.0"]["rank_prob"][a] for a in ag])
fig, ax = plt.subplots(figsize=(3.3, 3.2))
ax.imshow(P ** 0.6, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_yticks(range(len(ag))); ax.set_yticklabels(ag, fontsize=5.5)
ax.set_xticks(range(0, len(ag), 3)); ax.set_xticklabels(range(1, len(ag) + 1, 3), fontsize=6)
ax.set_xlabel("overall rank under item-noise resampling")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ranks.pdf"); plt.close(fig)
print("figs written:", OUT)
