#!/bin/bash

OUTPUT_BUNDLE_DIR="v0.4.0"
OUTPUT_SOURCE_ID="CR-CMIP-0-4-0"
UPLOAD_DIR="cr-cmip-0-4-0-1"
for variable_dir in output-bundles/"${OUTPUT_BUNDLE_DIR}"/data/processed/esgf-ready/input4MIPs/CMIP6Plus/CMIP/CR/"${OUTPUT_SOURCE_ID}"/atmos/yr/*; do
	echo "Uploading ${variable_dir}"
	pixi run input4mips-validation --logging-level INFO \
		upload-ftp \
		--ftp-dir-rel-to-root "${UPLOAD_DIR}" \
		--password zebedee.nicholls@climate-resource.com \
		--n-threads 6 \
		--cv-source "gh:main" \
		"${variable_dir}" || exit 1
done

for variable_dir in output-bundles/"${OUTPUT_BUNDLE_DIR}"/data/processed/esgf-ready/input4MIPs/CMIP6Plus/CMIP/CR/"${OUTPUT_SOURCE_ID}"/atmos/mon/*; do
	echo "Uploading $variable_dir"
	pixi run input4mips-validation --logging-level INFO \
		upload-ftp \
		--ftp-dir-rel-to-root "${UPLOAD_DIR}" \
		--password zebedee.nicholls@climate-resource.com \
		--n-threads 6 \
		--cv-source "gh:main" \
		"${variable_dir}" || exit 1
done
