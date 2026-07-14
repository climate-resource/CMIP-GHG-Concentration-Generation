#!/bin/bash
# Run helper
#
# Usage:
#
# Single gas for dev (here shown for CO2)
#   GAS="co2" RUN_ID="dev-test-run" bash run-helper.sh
#
# All gases for dev
#   GAS="all" RUN_ID="dev-test-run" bash run-helper.sh
#
# Single gas for specific output (here shown for CO2 for v0.3.0)
#   GAS="co2" RUN_ID="v0.3.0" bash run-helper.sh
#
# All gases for specific output (here shown for v0.3.0)
#   GAS="all" RUN_ID="v0.3.0" bash run-helper.sh

echo "RUN_ID=${RUN_ID}"
echo "GAS=${GAS}"

# Need to remake this, no matter what we're doing
rm -f dev-config-absolute.yaml && make dev-config-absolute.yaml

if [ "$RUN_ID" == "dev-test-run" ]; then

    doit_config_file="dev-config-absolute.yaml"

    if [ "$GAS" == "all" ]; then

        DOIT_CONFIGURATION_FILE=$doit_config_file DOIT_RUN_ID="dev-test-run" \
            DOIT_DB_BACKEND="json" DOIT_DB_FILE="doit-db-dev.json" \
            pixi run doit --verbosity=2 -n 4

    else

        # Run for individual gas
        DOIT_CONFIGURATION_FILE=$doit_config_file DOIT_RUN_ID="dev-test-run" \
            DOIT_DB_BACKEND="json" DOIT_DB_FILE="doit-db-dev.json" \
            pixi run doit --verbosity=2 -n 4 \
            "${PWD}/output-bundles/${RUN_ID}/data/processed/esgf-ready/${GAS}_input4MIPs_esgf-ready.complete"

    fi

else

    pixi run -e all-dev python scripts/write-run-config.py

    doit_config_file=$RUN_ID-config.yaml

    if [ "$GAS" == "all" ]; then

        DOIT_CONFIGURATION_FILE=$doit_config_file DOIT_RUN_ID="$RUN_ID" \
            pixi run doit run --verbosity=2 -n 4

        # Upload to zenodo
        pixi run python scripts/upload-bundle-to-zenodo.py "output-bundles/${RUN_ID}"

    else

        # Run for individual gas
        DOIT_CONFIGURATION_FILE=$doit_config_file DOIT_RUN_ID="$RUN_ID" \
            pixi run doit --verbosity=2 \
            "${PWD}/output-bundles/${RUN_ID}/data/processed/esgf-ready/${GAS}_input4MIPs_esgf-ready.complete"

    fi

fi
