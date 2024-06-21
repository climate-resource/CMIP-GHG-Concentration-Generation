#!/bin/bash

echo "${GAS}"
rm -f dev-config-absolute.yaml &&
	make dev-config-absolute.yaml &&
	# DOIT_CONFIGURATION_FILE=ci-config-absolute.yaml DOIT_RUN_ID="CI" DOIT_DB_FILE=".doit_ci" poetry run doit --verbosity=2 "${PWD}output-bundles/CI/data/processed/esgf-ready/${GAS}_input4MIPs_esgf-ready.complete" &&
	DOIT_CONFIGURATION_FILE=dev-config-absolute.yaml DOIT_RUN_ID="dev-test-run" DOIT_DB_BACKEND="json" DOIT_DB_FILE="doit-db-dev.json" poetry run doit --verbosity=2 "${PWD}/output-bundles/dev-test-run/data/processed/esgf-ready/${GAS}_input4MIPs_esgf-ready.complete"
