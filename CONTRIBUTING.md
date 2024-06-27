# Contributor Guide

## Development Installation
`pip install -e .[dev,test]`

## Style and Pre-Commit
To set up pre-commit:
```
pre-commit autoupdate
pre-commit install
```
Then any time you write a commit, it will run ruff, black, mypy, and validate the pyproject.toml. 
To skip the pre-commit step, use:
`git commit --no-verify`

Ruff, black, and mypy settings are specified in the pyproject.toml. Currently they are not very strict, but this may change in the future.

## Testing
To run tests with coverage locally:
`pytest tests --color=yes --cov --cov-report=term-missing`
This will also be run automatically when a PR is made to master and a codecov report will be generated telling you if your PR increased or decreased coverage.


## Branching and PRs
- Users that have been added to the CellMap organization and the DaCapo project should be able to develop directly into the CellMap fork of DaCapo. Other users will need to create a fork or ask to be added as a collaborator.
- For a completely new feature, make a branch off of the `main` branch of CellMap's fork of DaCapo with a name describing the feature. If you are collaborating on a feature that already has a branch, you can branch off that feature branch.
- Currently, you should make your PRs into the `main` branch of CellMap's fork, or the feature branch you branched off of. PRs currently require one maintainer's approval before merging. Once the PR is merged, the feature branch will/should be deleted. 
- `main` will be regularly published to PyPi when new features are fully implemented and all tests are passing.


## Documentation
Documentation is built using Sphinx. To build the documentation locally, run
```bash
sphinx-build -M html docs/source docs/build
```
This will generate the html files in the `docs/build/html` directory.
