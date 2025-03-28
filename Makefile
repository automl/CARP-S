# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

SHELL := /bin/bash

NAME := CARP-S
PACKAGE_NAME := carps
VERSION := 1.0.0

DIST := dist

.PHONY: help install-dev clean-build check build docs publish test

# TODO update help in Makefile
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
	$(PYTHON) -m pytest tests/test_configs.py tests/test_optimizers.py tests/test_tasks.py -n 8

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

uvenv:
	pip install uv
	uv venv --python=3.12 carpsenv
	. carpsenv/bin/activate
	uv pip install setuptools wheel

optimizer_smac:
	$(PIP) install swig
	$(PIP) install -r container_recipes/optimizers/SMAC3/SMAC3_requirements.txt

optimizer_optuna:
	$(PIP) install -r container_recipes/optimizers/Optuna/Optuna_requirements.txt

optimizer_dehb:
	$(PIP) install -r container_recipes/optimizers/DEHB/DEHB_requirements.txt
	$(PIP) install numpy --upgrade

optimizer_skopt:
	$(PIP) install -r container_recipes/optimizers/Scikit_Optimize/Scikit_Optimize_requirements.txt

optimizer_synetune:
	$(PIP) install -r container_recipes/optimizers/SyneTune/SyneTune_requirements.txt
	$(PIP) install numpy --upgrade

optimizer_ax:
	$(PIP) install -r container_recipes/optimizers/Ax/Ax_requirements.txt
	$(PIP) install numpy --upgrade

optimizer_hebo:
	# . container_recipes/optimizers/HEBO/HEBO_install.sh
	$(PIP) install -r container_recipes/optimizers/HEBO/HEBO_requirements.txt
	$(PIP) install numpy --upgrade

optimizer_nevergrad:
	$(PIP) install -r container_recipes/optimizers/Nevergrad/Nevergrad_requirements.txt
	$(PIP) install numpy --upgrade

benchmark_bbob:
	# Install BBOB
	$(PIP) install ioh
	$(PIP) install numpy --upgrade

benchmark_yahpo:
# Needs 2GB of space for the surrogate models of YAHPO
	# Install yahpo
	. container_recipes/benchmarks/YAHPO/install_yahpo.sh
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

benchmarks:
	$(MAKE) benchmark_bbob
	$(MAKE) benchmark_yahpo
	$(MAKE) benchmark_pymoo
	$(MAKE) benchmark_mfpbench
	$(MAKE) benchmark_hpobench
	$(MAKE) benchmark_hpob

optimizers:
	$(MAKE) optimizer_smac
	$(MAKE) optimizer_optuna
	$(MAKE) optimizer_dehb
	$(MAKE) optimizer_skopt
	$(MAKE) optimizer_synetune
	$(MAKE) optimizer_ax
	$(MAKE) optimizer_nevergrad
	# $(MAKE) optimizer_hebo

all:
	$(MAKE) benchmarks
	$(MAKE) optimizers

default:
	echo "Installed carps. If you want to install the benchmarks and optimizers, pass 'all' or e.g."
	echo "'benchmark_bbob optimizer_smac'"