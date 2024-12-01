#!/bin/bash

DOIT_CONFIGURATION_FILE=dev-config.yaml DOIT_RUN_ID="dev-test-run" DOIT_DB_BACKEND="json" DOIT_DB_FILE="doit-db-dev.json" pixi run -e all-dev doit list --all --quiet | while read line; do
	DOIT_CONFIGURATION_FILE=dev-config.yaml DOIT_RUN_ID="dev-test-run" DOIT_DB_BACKEND="json" DOIT_DB_FILE="doit-db-dev.json" pixi run -e all-dev doit info "${line}"
done
