.PHONY: check lint type test fmt

check: lint type test

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy src tests

test:
	pytest -q
