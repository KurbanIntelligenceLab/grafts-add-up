#!/bin/sh
# Four-pass build. No && chaining: every pass runs regardless of exit status,
# because pdflatex returns nonzero on warnings and a skipped pass leaves the
# PDF one generation behind the .aux (silent, and easy to miss).
set -x
pdflatex -interaction=nonstopmode main.tex > build/p1.log 2>&1
bibtex main                                > build/bib.log 2>&1
pdflatex -interaction=nonstopmode main.tex > build/p2.log 2>&1
pdflatex -interaction=nonstopmode main.tex > build/p3.log 2>&1
set +x
echo "--- undefined citations/references ---"
grep -icE "undefined (citation|reference)|multiply.defined" build/p3.log || true
echo "--- overfull boxes > 10pt ---"
grep -cE "Overfull \\\\hbox \([0-9]{2,}" build/p3.log || true
echo "--- pages ---"
pdfinfo main.pdf | grep Pages
