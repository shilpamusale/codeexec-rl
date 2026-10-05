.PHONY: check lint type test format

check: lint type test

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy src tests

test:
	pytest -q
