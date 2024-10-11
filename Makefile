install: ## Install the poetry environment and install the pre-commit hooks
	@echo "🚀 Install dependencies using Poetry"
	@poetry install

check: ## Run code quality tools.
	@echo "🚀 Checking Poetry lock file consistency with 'pyproject.toml': Running poetry lock --check"
	@poetry check --lock
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@poetry run deptry .

lint: ## Run linters
	@echo "🚀 Running Ruff lint"
	@ruff check . --fix
	

export: ## Export requirements.txt file
	@echo "🚀 Exporting requirements.txt file"
	@poetry export --without-hashes -f requirements.txt --output requirements.txt


build: clean-build ## Build wheel file using poetry
	@echo "🚀 Creating wheel file"
	@poetry build

clean-build: ## clean build artifacts
	@rm -rf dist

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help