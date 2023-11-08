.DEFAULT_GOAL := help

DEV_CONFIG_YAML=dev-config.yaml
DEV_CONFIG_ABSOLUTE_YAML=dev-config-absolute.yaml
DEV_RUN_ID="dev-test-run"
FINAL_DOIT_TASK="copy_source_into_output"
SHOW_CONFIGURATION_TASK="generate_workflow_tasks:Show configuration"

# A helper script to get short descriptions of each target in the Makefile
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([\$$\(\)a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-30s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:  ## print short description of each target
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


all:  ## compile all outputs
	# High verbosity for now, may split out `all` and `all-verbose` targets if verbosity is too annoying
	poetry run doit run --verbosity=2

doit-list:  ## list all the doit tasks
	poetry run doit list --all --status

all-dev:  ## compile all outputs using the dev run-id
	DOIT_CONFIGURATION_FILE=$(DEV_CONFIG_YAML) DOIT_RUN_ID=$(DEV_RUN_ID) poetry run doit run --verbosity=2

all-debug-dev:  ## compile all outputs using the dev run-id, falling to debugger on failure
	DOIT_CONFIGURATION_FILE=$(DEV_CONFIG_YAML) DOIT_RUN_ID=$(DEV_RUN_ID) poetry run doit run --pdb

doit-list-dev:  ## list all the doit tasks using the dev run-id
	DOIT_CONFIGURATION_FILE=$(DEV_CONFIG_YAML) DOIT_RUN_ID=$(DEV_RUN_ID) poetry run doit list --all --status

$(DEV_CONFIG_ABSOLUTE_YAML): $(DEV_CONFIG_YAML) scripts/create-dev-config-absolute.py
	poetry run python scripts/create-dev-config-absolute.py

.PHONY: checks
checks:  ## run all the linting checks of the codebase
	@echo "=== pre-commit ==="; poetry run pre-commit run --all-files || echo "--- pre-commit failed ---" >&2; \
		echo "=== mypy ==="; MYPYPATH=stubs poetry run mypy src notebooks || echo "--- mypy failed ---" >&2; \
		echo "======"


virtual-environment: pyproject.toml  ## update virtual environment, create a new one if it doesn't already exist
	# Put virtual environments in the project
	poetry config virtualenvs.in-project true
	poetry lock --no-update
	poetry install --all-extras
	poetry run pre-commit install
