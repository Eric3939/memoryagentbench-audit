"""Diff MemoryAgentBench per-system numbers between arXiv v1 and v4 for identically named columns.

Usage: python version_diff.py papers/pdf reports/data/mab_version_diff.csv
v1: main table (tables[1]) + appendix RULER table (tables[5]); v4: Table 3 (tables[3]).
"""
import csv, html, re, sys


def tables(path):
    t = open(path).read()
    out = []
    for tb in re.findall(r"<table.*?</table>", t, flags=re.S):
        rows = re.findall(r"<tr.*?</tr>", tb, flags=re.S)
        out.append([[html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                     for c in re.findall(r"<t[dh].*?</t[dh]>", r, flags=re.S)] for r in rows])
    return out


def grab(tab, hdr_row, cols):
    hdr = tab[hdr_row]
    res = {}
    for row in tab[hdr_row + 1:]:
        if len(row) != len(hdr) or row[0] in res:
            continue  # group-label rows; keep first of duplicated names
        res[row[0]] = {c: float(row[hdr.index(src)]) for c, src in cols.items() if src in hdr}
    return res


d = sys.argv[1]
t1, t4 = tables(f"{d}/mab_v1.html"), tables(f"{d}/mab_v4.html")
v1 = grab(t1[1], 1, {"LME(S*)": "LME(S*)", "EventQA": "EventQA", "MCC": "MCC", "Recom.": "Recom",
                      "Summ.": "∞\\inftyBench-Sum", "FC-SH": "FactCon-SH", "FC-MH": "FactCon-MH"})
for k, v in grab(t1[5], 0, {"SH-QA": "RULER-QA1", "MH-QA": "RULER-QA2"}).items():
    v1.setdefault(k, {}).update(v)
v4 = grab(t4[3], 1, {c: c for c in ["SH-QA", "MH-QA", "LME(S*)", "EventQA", "MCC", "Recom.", "Summ.", "FC-SH", "FC-MH"]})
rows = []
for sysname in v1:
    for alias in (sysname, sysname + " (128K)", sysname + " (200K)", sysname + " (1M)"):
        if alias in v4:
            for c, a in v1[sysname].items():
                if c in v4[alias]:
                    rows.append([sysname, c, a, v4[alias][c], round(v4[alias][c] - a, 1)])
            break
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["system", "column", "v1", "v4", "delta"]); w.writerows(rows)
ch = [r for r in rows if r[4] != 0]
print(f"{len(rows)} shared cells, {len(ch)} changed; |delta| >= 2: {sum(abs(r[4]) >= 2 for r in rows)}")
for r in sorted(ch, key=lambda r: -abs(r[4])):
    print(r)
