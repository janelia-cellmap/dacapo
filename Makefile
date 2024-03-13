default:
	pip install .

install-dev:
	pip install -e ".[dev]"

.PHONY: tests
tests:
	pytest -v --cov=dacapo dacapo
	flake8 dacapo
