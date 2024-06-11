# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

SHELL := /bin/bash

NAME := CARP-S
PACKAGE_NAME := carps
VERSION := 0.1.0

DIST := dist

.PHONY: help install-dev clean-build check build docs publish test

help:
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* clean-build      to clean any build files"
	@echo "* check            to run the pre-commit check"
	@echo "* build            to build a dist"
	@echo "* docs             to generate and view the html files, checks links"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python

test:
	$(PYTHON) -m pytest tests

docs:
	$(PYTHON) -m webbrowser -t "http://127.0.0.1:8000/"
	$(PYTHON) -m mkdocs serve --clean

check:
	pre-commit run --all-files

install-dev:
	pip install -e ".[dev]"
	pre-commit install

clean-build:
	rm -rf ${DIST}

# Build a distribution in ./dist
build:
	$(PYTHON) -m pip install build
	$(PYTHON) -m build --sdist

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean-build build
	read -p "Did you update the version number in Makefile, pyproject.toml, and CITATION.cff?"

	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uplaoded distribution into"
	@echo "* Run the following:"
	@echo "--- pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${PACKAGE_NAME}==${VERSION}"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo "--- python -c 'import ${PACKAGE_NAME}'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo "--- python -m twine upload dist/*"
