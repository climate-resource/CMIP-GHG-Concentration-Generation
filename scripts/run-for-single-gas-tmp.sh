#!/bin/bash

RUN_ID=20240729
echo "${GAS}"

rm -f dev-config-absolute.yaml && make dev-config-absolute.yaml
pixi run python scripts/write-run-config.py

if [ "$GAS" == "all" ]; then

	DOIT_CONFIGURATION_FILE=$RUN_ID-config.yaml DOIT_RUN_ID="$RUN_ID" \
		pixi run doit run --verbosity=2
	# -n 4

else

	# Run for individual gas
	DOIT_CONFIGURATION_FILE=$RUN_ID-config.yaml DOIT_RUN_ID="$RUN_ID" \
		pixi run doit --verbosity=2 \
		"${PWD}/output-bundles/${RUN_ID}/data/processed/esgf-ready/${GAS}_input4MIPs_esgf-ready.complete"

fi
