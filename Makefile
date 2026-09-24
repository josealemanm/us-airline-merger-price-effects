# Every stage, in order. `make all` runs the lot from nothing.
#
# The pull is the long pole: 5.2 GB over sixty quarterly files, about
# forty minutes on a normal connection. Everything after it runs in minutes.

PY := .venv/Scripts/python.exe
ifeq ($(wildcard $(PY)),)
PY := .venv/bin/python
endif

.PHONY: help setup all pull macro validate panel concentration did demand sim \
        screens figures dashboard memo test lint clean distclean

help:
	@echo "us-airline-merger-price-effects"
	@echo ""
	@echo "  make setup          create the virtual environment and install"
	@echo "  make all            run the whole pipeline end to end"
	@echo "  make test           run the test suite"
	@echo ""
	@echo "  stages, in order:"
	@echo "    pull              DOT DB1B, 60 quarters      (~40 min, 5.2 GB)"
	@echo "    macro             CPI-U and jet fuel, FRED   (~5 s)"
	@echo "    validate          data quality report"
	@echo "    panel             build the analysis panel"
	@echo "    concentration     HHI and the Guidelines screens"
	@echo "    did               the retrospective"
	@echo "    demand            demand estimation"
	@echo "    sim               merger simulation"
	@echo "    screens           score the predictions against outcomes"
	@echo "    figures           the charts"
	@echo "    dashboard         the browser dashboard"
	@echo "    memo              the one-page PDF"
	@echo ""
	@echo "  make clean          remove derived outputs, keep the raw pull"
	@echo "  make distclean      remove everything the pipeline produced"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

pull:          ; $(PY) src/01_pull_db1b.py
macro:         ; $(PY) src/02_pull_macro.py
validate:      ; $(PY) src/03_validate.py
panel:         ; $(PY) src/04_build_panel.py
concentration: ; $(PY) src/05_concentration.py
did:           ; $(PY) src/06_estimate_did.py
demand:        ; $(PY) src/07_estimate_demand.py
sim:           ; $(PY) src/08_merger_sim.py
screens:       ; $(PY) src/09_screens_vs_outcomes.py
figures:       ; $(PY) src/10_figures.py
dashboard:     ; $(PY) src/11_dashboard.py
memo:          ; $(PY) src/12_memo.py

all: pull macro validate panel concentration did demand sim screens figures \
     dashboard memo
	@echo ""
	@echo "done. reports/ holds the write-ups, docs/dashboard.html the browser view."

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf data/processed/*.parquet data/processed/*.json
	rm -rf reports/figures/*.png reports/*.md reports/*.pdf
	rm -rf data/interim/*.parquet

distclean: clean
	rm -rf data/raw/*.parquet data/raw/manifest.json data/raw/_scratch
