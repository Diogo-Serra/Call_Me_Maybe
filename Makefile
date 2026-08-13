# Makefile for Call_me_maybe

ENV_PIP := .venv/bin/pip
ENV_UV := .venv/bin/uv
MYPY_FLAGS := --warn-return-any --warn-unused-ignores \
			  --ignore-missing-imports --disallow-untyped-defs \
			  --check-untyped-defs

install:
	@echo Installing dependecies and preparing the system.
	python3 -m venv .venv
	$(ENV_PIP) install uv
	$(ENV_UV) sync

run:
	$(ENV_UV) run python -m src

debug:
	$(ENV_UV) run python -m pdb -m src

lint:
	$(ENV_UV) run flake8 .
	$(ENV_UV) run mypy . $(MYPY_FLAGS)

lint-strict:
	$(ENV_UV) run flake8 .
	$(ENV_UV) run mypy . --strict

clean:
	@echo Cleaning environment ...
	rm -rf .venv
	rm -rf src/data/output
	rm -rf $$(find . -name __pycache__ -o -name .mypy_cache)
