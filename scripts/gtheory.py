"""Variance components for one system on one dataset: fully crossed item × rebuild × order design.

y[i, k, o] in {0,1} (or [0,1]) = correctness of item i under memory rebuild k and ingestion order o.
Random-effects ANOVA (method of moments, balanced design; negative estimates truncated at 0):
    y = mu + a_i + b_k + c_o + (ab)_ik + (ac)_io + (bc)_ko + e_iko
The quantity that matters for leaderboards is the spread of a *single published score* (one build, one
order, fixed item set): SD_run = sqrt(s2_k + s2_o + s2_ko + (s2_ik + s2_io + s2_e) / n_items).
Item-sampling SD of a score (generalising over items) = sqrt((s2_i + s2_ik + s2_io + s2_e) / n_items).

`python gtheory.py simulate` runs a power study: how well K rebuilds × O orders recover SD_run.
"""
import sys
import numpy as np


def components(y):
    y = np.asarray(y, float)
    n, K, O = y.shape
    m = y.mean()
    mi, mk, mo = y.mean((1, 2)), y.mean((0, 2)), y.mean((0, 1))
    mik, mio, mko = y.mean(2), y.mean(1), y.mean(0)
    ss = {
        "i": K * O * ((mi - m) ** 2).sum(), "k": n * O * ((mk - m) ** 2).sum(), "o": n * K * ((mo - m) ** 2).sum(),
        "ik": O * ((mik - mi[:, None] - mk[None, :] + m) ** 2).sum(),
        "io": K * ((mio - mi[:, None] - mo[None, :] + m) ** 2).sum(),
        "ko": n * ((mko - mk[:, None] - mo[None, :] + m) ** 2).sum(),
    }
    resid = y - mik[:, :, None] - mio[:, None, :] - mko[None, :, :] + mi[:, None, None] + mk[None, :, None] + mo[None, None, :] - m
    ss["e"] = (resid ** 2).sum()
    df = {"i": n - 1, "k": K - 1, "o": O - 1, "ik": (n - 1) * (K - 1), "io": (n - 1) * (O - 1),
          "ko": (K - 1) * (O - 1), "e": (n - 1) * (K - 1) * (O - 1)}
    ms = {f: ss[f] / df[f] if df[f] > 0 else 0.0 for f in ss}
    s2 = {"e": ms["e"]}
    s2["ik"] = (ms["ik"] - ms["e"]) / O
    s2["io"] = (ms["io"] - ms["e"]) / K
    s2["ko"] = (ms["ko"] - ms["e"]) / n
    s2["i"] = (ms["i"] - ms["ik"] - ms["io"] + ms["e"]) / (K * O)
    s2["k"] = (ms["k"] - ms["ik"] - ms["ko"] + ms["e"]) / (n * O)
    s2["o"] = (ms["o"] - ms["io"] - ms["ko"] + ms["e"]) / (n * K)
    s2 = {f: max(v, 0.0) for f, v in s2.items()}
    sd_run = np.sqrt(s2["k"] + s2["o"] + s2["ko"] + (s2["ik"] + s2["io"] + s2["e"]) / n)
    sd_item = np.sqrt((s2["i"] + s2["ik"] + s2["io"] + s2["e"]) / n)
    ever_flip = float(((y.max((1, 2)) - y.min((1, 2))) > 0).mean())
    return {"s2": s2, "sd_run": float(sd_run), "sd_item": float(sd_item), "mean": float(m), "ever_flip": ever_flip,
            "score_range": float(y.mean(0).max() - y.mean(0).min())}


def simulate_once(rng, n, K, O, sd_build, sd_order, item_flip):
    """Logit model: item difficulty ~ N(0, 1.5); whole-build shift ~ N(0, sd_build); order shift ~ N(0, sd_order);
    plus per-(item, build, order) logit noise item_flip (item-level instability)."""
    a = rng.normal(0, 1.5, n)
    b = rng.normal(0, sd_build, K)
    c = rng.normal(0, sd_order, O)
    z = a[:, None, None] + b[None, :, None] + c[None, None, :] + rng.normal(0, item_flip, (n, K, O))
    return (rng.random((n, K, O)) < 1 / (1 + np.exp(-z))).astype(float)


def true_sd_run(n, sd_build, sd_order, item_flip, rng, reps=400):
    """Monte Carlo SD of single-cell scores across fresh builds/orders for one fixed item set."""
    a = rng.normal(0, 1.5, n)
    s = []
    for _ in range(reps):
        z = a + rng.normal(0, sd_build) + rng.normal(0, sd_order) + rng.normal(0, item_flip, n)
        s.append((rng.random(n) < 1 / (1 + np.exp(-z))).mean())
    return float(np.std(s))


def simulate():
    rng = np.random.default_rng(0)
    n = 100
    print("scenario (logit SDs)          K  O   true SD_run  est. median [5%,95%]   (points)")
    for sd_b, sd_o, flip in [(0.0, 0.0, 0.0), (0.3, 0.2, 0.5), (0.6, 0.4, 1.0)]:
        truth = 100 * true_sd_run(n, sd_b, sd_o, flip, rng)
        for K, O in [(3, 3), (5, 5), (10, 3)]:
            est = [100 * components(simulate_once(rng, n, K, O, sd_b, sd_o, flip))["sd_run"] for _ in range(300)]
            q = np.percentile(est, [5, 50, 95])
            print(f"build {sd_b:.1f} order {sd_o:.1f} flip {flip:.1f}   {K:2d} {O:2d}   {truth:6.2f}      "
                  f"{q[1]:5.2f} [{q[0]:5.2f}, {q[2]:5.2f}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        simulate()
