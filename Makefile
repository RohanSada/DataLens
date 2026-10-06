.PHONY: install dev lint test smoke demo-db serve report

install:        ## core + serving + evaluation
	pip install -e ".[serve,eval]"

dev:            ## everything for development
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	pytest -m "not smoke"

smoke:          ## SFT + GRPO on a tiny CPU model (needs the [train] extra)
	pytest -m smoke

demo-db:
	python scripts/make_demo_db.py data/databases

serve: demo-db
	datalens serve

report:         ## compare every finished run
	datalens eval report runs/* --out reports/latest
