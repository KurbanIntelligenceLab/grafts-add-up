# Clean-room paper bundle

Compile from `paper/` (the Overleaf folder). Figure inputs are
`paper/figs/fig*.tex` and `paper/figs/*.dat`. Do not expect a
sibling `figs/` directory inside an unpacked zip.

## Files that must be in the zip

- `main.tex`
- `suture.bib`
- `build.sh`
- `iclr2027_conference.sty`
- `iclr2027_conference.bst`
- `figs/` (all `.tex` and `.dat` files in `paper/figs/`)

## Files that must not be in the paper zip

- `results/` and adapter `*.safetensors`
- Hugging Face snapshots
- `__pycache__/`
- `env.lock` (local, bloated; use `requirements-gpu.txt` in the code release)
- `third_party/iclr2027/` template extras (`natbib.sty`, `fancyhdr.sty`, `math_commands.tex`)
- `third_party/iclr2027/iclr-2027-style-files.zip`
- Internal author-process files: advisor, human-gate, contract-review, and
  novelty-search notes
- Local review notes and experiment code; the paper zip is compile-only

The anonymous OpenReview PDF zip is **only** the compile set listed above.
Code supplementary uses `requirements-gpu.txt`, not `env.lock`.

## Build

```
cd paper
./build.sh
```

Requires `pdflatex`, `bibtex`, and `pdfinfo`. On Windows, run the same four
passes from `paper/` if `build.sh` is unavailable:

```
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`paper/` already uses the official ICLR 2027 class. Re-check the page
count after any preamble change. Main text must fit in 9 pages; the appendix
may continue after that.

## Verify from an empty directory

1. Copy only the files listed above into a fresh folder.
2. Run the four-pass build there.
3. Confirm `main.pdf` exists and that the log has no undefined citations or
   references.
