.PHONY: check lint type test format

check: lint type test

lint:
	ruff check src tests
	ruff format --check src tests
	shellcheck scripts/*.sh

format:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy src tests

test:
	pytest -q -m "not integration and not dataset"
