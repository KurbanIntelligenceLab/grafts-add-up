# Paper build notes (16 September 2026, post powered Spanish ranking)

Toolchain: MiKTeX 25.12, `pdflatex` + `bibtex`, official
`iclr2027_conference.sty` / `.bst`.

## Layout

- Total: 24 pages
- Main text (counts): pages 1--9. Section 8 Limitations finishes on page 9.
- Reproducibility, ethics, and AI-use: page 10 (do not count)
- Bibliography: pages 11--13
- Appendix: pages 14--24

## Agent visual pass of page 9

Page 9 contains the last protocol sentence, all of Related work, and all of
Limitations. Reproducibility starts on page 10. The official style and
bibliography hashes match the downloaded ICLR 2027 files; no duplicate line
number ruler is present.

## Log inspection

- Undefined citations / references: none
- Overfull `\hbox` of 10pt or more: none
- Underfull boxes remain around long appendix identifiers; they do not overrun
  the margins.
- ACM / Expert Merging remain arXiv preprints

## Human still required

Named-author proof signature and 2--3 independent mock reviews.
See `HUMAN_GATES.md`. Author-led freeze is already recorded; do not wait
for another advisor email.

## B3 integration

The main text reports the bounded French/Chinese B3 candidate comparison. The
power table was moved to Appendix G while retaining its values and references,
so the ICLR main-text limit remains satisfied. The four-pass render after the
64-item ranking disclosure has no undefined citations, LaTeX errors, overfull
boxes, or unresolved references. Limitations still finishes on page 9.

## Expanded B3 disposition

The separate five-language follow-up is not part of this build. It stopped at
the German generation pilot when the frozen BF16 cache-equivalence gate found
different cached and uncached greedy outputs. No complete panel manifest or
post-hoc statistics were produced, and no expanded number is included in the
paper. The canonical v1 B3 artifacts and manuscript claims are unchanged.

## Powered Spanish ranking

The isolated 64-item greedy Coder ranking finished 16 September 2026
(20.7 GPU-hours, 406/406 windows). Spearman 0.129; selected `[1,27]` at
1/64 exact match versus sweep best 2/64. Reported in Section 6, Appendix D,
the Appendix G power table, and Appendix I. Canonical E1 was not rewritten.
Main text still ends on page 9 after the Section 6 and Limitations updates.
