# Top-level Makefile — orchestrates the v5.0.0 cross-platform paper bench.
#
# Usage:
#   make bench-darwin-arm64 WAVE=1
#   make bench-all          WAVE=1
#   make bench-arxiv        WAVE=1

WAVE ?= 1

.PHONY: bench-all bench-darwin-arm64 bench-darwin-x86 bench-linux-x64 bench-windows bench-arxiv aggregate

bench-all: bench-darwin-arm64 bench-linux-x64
	$(MAKE) aggregate

bench-darwin-arm64:
	./reproduce_paper.sh --platform darwin-arm64 --wave $(WAVE) --runs 3

bench-darwin-x86:
	./reproduce_paper.sh --platform darwin-x86_64 --wave $(WAVE) --runs 3

bench-linux-x64:
	./reproduce_paper.sh --platform linux-x86_64 --wave $(WAVE) --runs 3

bench-windows:
	powershell -File reproduce_paper.ps1 -Platform windows-x86_64 -Wave $(WAVE) -Runs 3

aggregate:
	python scripts/aggregate_paper_tables.py results/paper/*.json

# Bench-arxiv: render the paper notebook to PDF after collecting bench data.
# `nbconvert --execute` re-runs every cell, picking up the JSON we just wrote.
bench-arxiv: bench-all
	jupyter nbconvert --to pdf notebooks/vectro_paper_results.ipynb --execute
