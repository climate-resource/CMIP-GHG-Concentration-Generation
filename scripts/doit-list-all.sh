#!/bin/bash

DOIT_CONFIGURATION_FILE=dev-config.yaml DOIT_RUN_ID="dev-test-run" DOIT_DB_BACKEND="json" DOIT_DB_FILE="doit-db-dev.json" pixi run -e all-dev doit list --all --quiet | while read line; do
	# Could then parse this output to create a table of dependencies and
	# targets, although that must already exist in doit somewhere internally...
	DOIT_CONFIGURATION_FILE=dev-config.yaml DOIT_RUN_ID="dev-test-run" DOIT_DB_BACKEND="json" DOIT_DB_FILE="doit-db-dev.json" pixi run -e all-dev doit info "${line}"
done
