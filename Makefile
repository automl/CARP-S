# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

SHELL := /bin/bash

NAME := CARP-S
PACKAGE_NAME := carps
VERSION := 0.1.1

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
PIP ?= pip

test:
	$(PYTHON) -m pytest tests

docs:
	$(PYTHON) -m webbrowser -t "http://127.0.0.1:8000/"
	$(PYTHON) -m mkdocs serve --clean

check:
	pre-commit run --all-files

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

clean-build:
	rm -rf ${DIST}

# Build a distribution in ./dist
build:
	$(PYTHON) -m $(PIP) install build
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
	@echo "* Create a new virtual environment to install the uploaded distribution into"
	@echo "* Run the following:"
	@echo "--- pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${PACKAGE_NAME}==${VERSION}"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo "--- python -c 'import ${PACKAGE_NAME}'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo "--- python -m twine upload dist/*"


OS := $(shell uname)

install:
	pip install uv
	uv venv --python=3.12 env_smbm
	. env_smbm/bin/activate
	$(MAKE) install-swig
	uv pip install setuptools wheel
	git clone --branch $(SMACBRANCH) git@github.com:automl/SMAC3.git repos/SMAC3
	git clone git@github.com:automl/CARP-S.git repos/CARP-S
	cd repos/CARP-S && uv pip install -e '.[dev]' && pre-commit install
	cd repos/SMAC3 && uv pip install -e '.[dev]' && pre-commit install
	$(MAKE) benchmark_bbob
	# $(MAKE) benchmark_yahpo
	$(MAKE) welcome

install-swig:
	@if [ "$(OS)" = "Darwin" ]; then \
		echo "Detected macOS: Installing SWIG via Homebrew"; \
		brew install swig; \
	elif [ "$(OS)" = "Linux" ]; then \
		echo "Detected Linux: Installing SWIG via APT"; \
		# sudo apt update && sudo apt install -y swig; \
		uv pip install swig; \
	elif [ "$(OS)" = "Windows_NT" ]; then \
		echo "Detected Windows: Installing SWIG via Chocolatey"; \
		choco install swig -y; \
	else \
		echo "Unsupported OS: Please install SWIG manually."; \
	fi

optimizer_smac:
	$(MAKE) install-swig
	$(PIP) install -r container_recipes/optimizers/SMAC3/SMAC3_requirements.txt

benchmark_bbob:
	# Install BBOB
	$(PIP) install ioh
	$(PIP) install numpy --upgrade

benchmark_yahpo:
# Needs 2GB of space for the surrogate models of YAHPO
	# Install yahpo
	CARPS_ROOT="." . container_recipes/benchmarks/YAHPO/install_yahpo.sh
	$(PIP) install ConfigSpace --upgrade

benchmark_pymoo:
	# Install pymoo
	$(PIP) install -r container_recipes/benchmarks/Pymoo/Pymoo_requirements.txt

benchmark_mfpbench:
	# Install mfpbench
	$(PIP) install -r container_recipes/benchmarks/MFPBench/MFPBench_requirements.txt
	$(PIP) install ConfigSpace --upgrade
	. container_recipes/benchmarks/MFPBench/download_data.sh

benchmark_hpobench:
	# Install hpobench
	. container_recipes/benchmarks/HPOBench/install_HPOBench.sh

benchmark_hpob:
	# Install hpob
	$(PIP) install -r container_recipes/benchmarks/HPOB/HPOB_requirements.txt
	. container_recipes/benchmarks/HPOB/download_data.sh

