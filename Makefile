# Makefile for rag-chatbot (Python)

# Project info
APP_NAME := rag-chatbot
PYPROJECT := pyproject.toml
SRC_DIR := .
DIST_DIR := dist
COVERAGE_DIR := coverage
VENV_DIR := .venv

# Tooling
UV := uv
UVX := uvx
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

.PHONY: help install lock build format lint fix deptry vuln-check bandit run clean clean-pyc clean-dist clean-venv clean-all check prepare ensure-uv
.DEFAULT_GOAL := help

# --- Helpers ---
define require_uv
    @command -v $(UV) >/dev/null 2>&1 || { \
        echo "$(RED)uv not found. Install uv: https://docs.astral.sh/uv/getting-started/$(NC)"; \
        exit 1; \
    }
endef

define in_venv
    VIRTUAL_ENV=$(VENV_DIR) $(1)
endef

# --- Help ---
help:
	@echo "$(BLUE)RAG Chatbot - Available targets:$(NC)"
	@echo "$(GREEN)Environment:$(NC)"
	@printf "  %-16s %s\n" "install" "Install all deps incl. dev (locked if uv.lock)"
	@printf "  %-16s %s\n" "lock" "Update uv.lock from pyproject"
	@echo "$(GREEN)Quality:$(NC)"
	@printf "  %-16s %s\n" "format" "Format code via ruff"
	@printf "  %-16s %s\n" "lint" "Lint via ruff (E,F,S,B)"
	@printf "  %-16s %s\n" "fix" "Auto-fix (ruff E,F,S,B)"
	@printf "  %-16s %s\n" "deptry" "Check (un)used deps via deptry"
	@echo "$(GREEN)Build:$(NC)"
	@printf "  %-16s %s\n" "build" "Build sdist and wheel into dist/"
	@echo "$(GREEN)Run:$(NC)"
	@printf "  %-16s %s\n" "run" "Run local app (python main.py)"
	@echo "$(GREEN)Clean:$(NC)"
	@printf "  %-16s %s\n" "clean" "Clean caches, dist, venv (safe)"
	@printf "  %-16s %s\n" "clean-all" "Deep clean everything"
	@echo "$(GREEN)Meta:$(NC)"
	@printf "  %-16s %s\n" "check" "format + lint + deptry"
	@printf "  %-16s %s\n" "prepare" "format + lint + build + clean"

# --- Dependency management ---

install: ensure-uv
	@echo "$(BLUE)Syncing runtime dependencies with uv...$(NC)"
	$(call in_venv,$(UV) sync --locked --dev)
	@echo "$(GREEN)Dependencies installed in $(VENV_DIR)$(NC)"

lock: ensure-uv
	@echo "$(BLUE)Updating uv.lock from $(PYPROJECT)...$(NC)"
	$(UV) lock
	@echo "$(GREEN)uv.lock updated$(NC)"

ensure-uv:
	$(call require_uv)

# --- Code quality ---
format: install
	@echo "$(BLUE)Formatting with ruff...$(NC)"
	$(call in_venv,$(UV) run -q -m ruff format $(SRC_DIR) || true)
	@echo "$(GREEN)Formatting complete$(NC)"

lint: install
	@echo "$(BLUE)Linting with ruff (E,F,S,B)...$(NC)"
	$(call in_venv,$(UV) run -q -m ruff check $(SRC_DIR))

fix: install
	@echo "$(BLUE)Auto-fixing with ruff (E,F,S,B)...$(NC)"
	$(call in_venv,$(UV) run -q -m ruff check --fix $(SRC_DIR) || true)

deptry: install
	@echo "$(BLUE)Checking dependencies with deptry...$(NC)"
	$(call in_venv,$(UV) run -q -m deptry $(SRC_DIR))

# --- Build ---
build: ensure-uv
	@echo "$(BLUE)Building package (sdist + wheel) with uv into $(DIST_DIR)...$(NC)"
	$(call in_venv,$(UV) build)
	@echo "$(GREEN)Build complete -> $(DIST_DIR)$(NC)"

# --- Run ---
run: install
	@echo "$(BLUE)Starting $(APP_NAME)...$(NC)"
	$(call in_venv,$(UV) run -q python main.py)

# --- Aggregate checks ---
check: format lint deptry
	@echo "$(GREEN)All checks completed$(NC)"

prepare: format lint build clean
	@echo "$(GREEN)Prepare completed: formatted, secured, built$(NC)"

# --- Cleanup ---
clean-pyc:
	@echo "$(BLUE)Cleaning Python caches...$(NC)"
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-dist:
	@echo "$(BLUE)Cleaning distribution artifacts...$(NC)"
	@rm -rf $(DIST_DIR) $(COVERAGE_DIR) rag_chatbot.egg-info

clean-venv:
	@echo "$(BLUE)Removing virtual environment $(VENV_DIR)...$(NC)"
	@rm -rf $(VENV_DIR)

clean: clean-pyc clean-dist
	@echo "$(GREEN)Clean complete$(NC)"

clean-all: clean clean-venv
	@echo "$(GREEN)Deep clean complete$(NC)"
