#!/bin/bash

OUTPUT_BUNDLE_DIR="v0.3.0"
for variable_dir in output-bundles/"${OUTPUT_BUNDLE_DIR}"/data/processed/esgf-ready/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-3-0/atmos/yr/*; do
	echo "Uploading ${variable_dir}"
	pixi run input4mips-validation --logging-level INFO \
		upload-ftp \
		--ftp-dir-rel-to-root cr-cmip-0-3-0-2 \
		--password zebedee.nicholls@climate-resource.com \
		--n-threads 6 \
		--cv-source "gh:cr-cmip-0-3-0" \
		"${variable_dir}" || exit 1
done

for variable_dir in output-bundles/"${OUTPUT_BUNDLE_DIR}"/data/processed/esgf-ready/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-0-3-0/atmos/mon/*; do
	echo "$variable_dir"
	pixi run input4mips-validation --logging-level INFO \
		upload-ftp \
		--ftp-dir-rel-to-root cr-cmip-0-3-0-2 \
		--password zebedee.nicholls@climate-resource.com \
		--n-threads 6 \
		--cv-source "gh:cr-cmip-0-3-0" \
		"${variable_dir}" || exit 1
done
