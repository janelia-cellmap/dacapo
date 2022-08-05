default:
	pip install .

install-dev:
	pip install -r requirements.txt
	pip install -e .

.PHONY: tests
tests:
	pytest -v --cov=dacapo dacapo
	flake8 dacapo
