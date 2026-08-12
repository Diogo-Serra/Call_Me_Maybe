# Makefile for Call_me_maybe

ENV_PY := .venv/bin/python3
ENV_PIP := .venv/bin/pip
ENV_UV := .venv/bin/uv

install:
	python3 -m venv .venv
	$(ENV_PIP) install uv
	$(ENV_UV) sync

run:
	$(ENV_UV) run test_1.py
