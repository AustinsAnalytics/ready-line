# Ready Line - rebuild everything from public data.
#   make all      full pipeline (about 45 minutes on an M2 laptop; most of it is stage 10)
#   make site     rebuild only the web page from existing results
#   make serve    preview the site at http://localhost:8000
PY ?= python

.PHONY: all data etl warehouse explore test models nlp forecast simulate optimize site serve clean

all: data etl warehouse explore test models nlp forecast simulate optimize site

data:
	$(PY) pipeline/00_download.py
etl:
	$(PY) pipeline/01_spark_etl.py
warehouse:
	$(PY) pipeline/02_warehouse.py
explore:
	$(PY) pipeline/03_explore.py
test:
	$(PY) pipeline/04_hypotheses.py
models:
	$(PY) pipeline/05_rul_gbm.py
	$(PY) pipeline/06_rul_cnn.py
	$(PY) pipeline/07_evaluate.py
nlp:
	$(PY) pipeline/08_nlp_sdr.py
forecast:
	$(PY) pipeline/09_forecast.py
simulate:
	$(PY) pipeline/10_simulate.py
	$(PY) pipeline/10b_board.py
optimize:
	$(PY) pipeline/11_optimize.py
site:
	node pipeline/validate_js_sim.js
	$(PY) pipeline/12_build_site.py

serve:
	$(PY) -m http.server 8000 --directory site

clean:
	rm -rf data/processed data/warehouse.duckdb
