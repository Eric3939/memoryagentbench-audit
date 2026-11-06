# Paper draft (AES short paper)

- `main.tex` — draft; Part A (claims/noise audit, drift, reproductions) written from real analyses; `\tbd{}` (red)
  marks Part B placeholders and missing author info. `related_work.tex` is the long version; `related_work_short.tex` is the one used.
- `figs/` — regenerate with `python scripts/make_figs.py .` after updating the analyses.
- **Compile on Overleaf with XeLaTeX** (upload this folder; template from github.com/agent-evalscience/AES-Latex-Template).
  Local tectonic (TeX snapshot too old for aes.sty's hyperref patches) fails; a TeX Live 2025+ install would work.
- `references.bib` — 37 entries from arXiv metadata; check venues before submission.
- Static checks (2026-10-15 05:15): all \cite keys in references.bib, all \ref labelled, braces/environments balanced.
  Length ≈ 2,400 words main text + ≈ 1,100 related work + 2 figures + 2 tables → likely > 6 pp at 11pt; trim related
  work to ~500 words and move the MDD table or rank heatmap to the appendix (appendices don't count).
- Tables are generated: `scripts/make_harness_table.py` → harness_table.tex, `scripts/resolution_limits.py` → resolution_table.tex.
