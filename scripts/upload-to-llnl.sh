#!/bin/bash

OUTPUT_BUNDLE_DIR="v1.0.0"
OUTPUT_SOURCE_ID="CR-CMIP-1-0-0"
UPLOAD_DIR="cr-cmip-1-0-0-1"
# Depressing but seems to be all the server can handle
N_THREADS=1
LOG_LEVEL="INFO"
# LOG_LEVEL="DEBUG"

for variable_dir in output-bundles/"${OUTPUT_BUNDLE_DIR}"/data/processed/esgf-ready/input4MIPs/CMIP7/CMIP/CR/"${OUTPUT_SOURCE_ID}"/atmos/yr/*; do
    echo "Uploading ${variable_dir}"

    pixi run input4mips-validation --logging-level "${LOG_LEVEL}" \
        upload-ftp \
        --ftp-dir-rel-to-root "${UPLOAD_DIR}" \
        --password zebedee.nicholls@climate-resource.com \
        --n-threads $N_THREADS \
        --cv-source "gh:main" \
        "${variable_dir}" || exit 1
done

for variable_dir in output-bundles/"${OUTPUT_BUNDLE_DIR}"/data/processed/esgf-ready/input4MIPs/CMIP7/CMIP/CR/"${OUTPUT_SOURCE_ID}"/atmos/mon/*; do
    echo "Uploading $variable_dir"

    # # Oh what fun this was
    # if [[ $variable_dir == *ccl4 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cf4 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *c2f6 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *c3f8 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *c4f10 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *c5f12 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *c6f14 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *c7f16 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *c8f18 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cc4f8 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cfc11 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cfc113 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cfc114 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cfc115 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cfc11eq ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cfc12 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *cfc12eq ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *ch2cl2 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *ch3br ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *ch3ccl3 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *ch3cl ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *ch4 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *chcl3 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *co2 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *halon1211 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *halon1301 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *halon2402 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *hcc140a ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *hcfc141b ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *hcfc142b ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *hcfc22 ]]; then
    #     continue
    # fi
    # if [[ $variable_dir == *hfc125 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc134a ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc134aeq ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc143a ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc152a ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc227ea ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc23 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc236fa ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc245fa ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc32 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc365mfc ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *hfc4310mee ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *n2o ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *nf3 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *pfc116 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *pfc218 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *pfc318 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *pfc3110 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *pfc4112 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *pfc5114 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *pfc6116 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *pfc7118 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *sf6 ]]; then
    # 	continue
    # fi
    # if [[ $variable_dir == *so2f2 ]]; then
    # 	continue
    # fi

    pixi run input4mips-validation --logging-level "${LOG_LEVEL}" \
        upload-ftp \
        --ftp-dir-rel-to-root "${UPLOAD_DIR}" \
        --password zebedee.nicholls@climate-resource.com \
        --n-threads $N_THREADS \
        --cv-source "gh:main" \
        "${variable_dir}" || exit 1

    echo "Sleep"
    sleep 1

done
