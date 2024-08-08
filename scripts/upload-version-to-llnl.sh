#!/bin/bash
RUN_ID=20240729

pixi run input4mips-validation --logging-level INFO_FILE \
	upload-ftp \
	output-bundles/$RUN_ID/data/processed/esgf-ready/input4MIPs/ \
	--ftp-dir-rel-to-root "cr-testing-101" \
	--password "zebedee.nicholls@climate-resource.com" \
	--n-threads 8 \
	--cv-source "gh:main"
