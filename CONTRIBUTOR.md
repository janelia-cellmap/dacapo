# Contributor Guide

## Development Installation
- currently, pip install -r requirements-dev.txt
and then pip install -e .
TODO: set up pip install -e .[dev]

## Style and Pre-Commit
- TODO: pre-commit with black mypy
- do use ruff and black
- do have typing, developers should supply types

## Testing
- Unittest your functions
- Pytest run automatically on PR, and so is codcov (maybe)
TODO: codecov


## Branching and PRs
- Users that have been added to the CellMap organization and the DaCapo project should be able to develop directly into the CellMap fork of DaCapo. Other users will need to create a fork.
- For a completely new feature, make a branch off of the `main` branch of CellMap's fork of DaCapo with a name describing the feature. If you are collaborating on a feature that already has a branch, you can branch off that feature branch.
- Currently, you should make your PRs into the main branch of CellMap's fork, or the feature branch you branched off of. Once the PR is merged, the feature branch should be deleted. 
