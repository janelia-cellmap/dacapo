default:
	pip install .

install-dev:
	pip install -e .
	pip install --upgrade -r requirements-dev.txt

.PHONY: tests
tests:
	pytest -v --cov=dacapo dacapo
	flake8 dacapo
