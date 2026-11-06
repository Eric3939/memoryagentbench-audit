"""Write paper/harness_table.tex from reports/data/harness_findings.json (severity x evidence table)."""
import json
d = json.load(open('reports/data/harness_findings.json'))
ver = {"A1": "executed (real Mem0)", "A2": "stubbed execution", "A3": "code reading", "B1": "measured", "B2": "measured",
       "B3": "code + measured", "B4": "code reading", "C1": "offline run", "C2": "code + README", "C3": "code reading",
       "D1": "configs (32/32)", "E1": "code reading"}
eff = {"A1": "no retrieval", "A2": "cross-context leakage", "A3": "likely crash / empty", "B1": "evidence lost",
       "B2": "evidence lost", "B3": "evidence lost", "B4": "fewer passages", "C1": "false negatives",
       "C2": "false negatives", "C3": "false positives", "D1": "unreported variance", "E1": "judge drift"}


def tex(x):
    for a, b in [('\\', '\\textbackslash{}'), ('_', '\\_'), ('%', '\\%'), ('&', '\\&'), ('→', '$\\rightarrow$'),
                 ('≈', '$\\approx$'), ('–', '--'), ("'", "'")]:
        x = x.replace(a, b)
    return x


rows = []
for f in d['findings']:
    k = list(f['affects'])
    aff = ", ".join(k) if len(k) <= 3 else f"{len(k)} systems"
    rows.append(" & ".join([f['id'], tex(f['title']), tex(aff), eff[f['id']], ver[f['id']]]) + " \\\\")
out = ("\\begin{table}[t]\n\\centering\\footnotesize\n"
       "\\begin{tabularx}{\\linewidth}{@{}l X p{2.2cm} p{2.0cm} p{2.0cm}@{}}\n\\toprule\n"
       "ID & Finding (public harness, commit 5380260) & Affects & Effect & Evidence \\\\\n\\midrule\n"
       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n"
       "\\caption{Harness findings by effect and strength of evidence. Effects on the \\emph{published} MAB numbers are "
       "not established: the original runs may have used different code (Section~\\ref{sec:harness}).}\n"
       "\\label{tab:harness}\n\\end{table}\n")
open('paper/harness_table.tex', 'w').write(out)
print(out[:500])
