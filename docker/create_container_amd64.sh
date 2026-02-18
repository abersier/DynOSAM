#!/usr/bin/env bash

CONTAINER_NAME=dyno_pipeline
CONTAINER_IMAGE_NAME=acfr_rpg/dyno_pipeline_cuda
DYNOSAM_ROOT=$HOME/Documents/Thesis/dyno_pipeline

### EDIT THIS TO WHEREVER YOU'RE STORING YOU DATA ###
# folder should exist before you mount it
LOCAL_DATA_FOLDER=$DYNOSAM_ROOT/data/datasets/
LOCAL_RESULTS_FOLDER=$DYNOSAM_ROOT/data/results/
LOCAL_CORE_DYNO_SAM_FOLDER=$DYNOSAM_ROOT/code/core/
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=$DYNOSAM_ROOT/code/extras/

bash create_container_base.sh acfr_rpg/dyno_pipeline_cuda dyno_pipeline $LOCAL_DATA_FOLDER $LOCAL_RESULTS_FOLDER $LOCAL_CORE_DYNO_SAM_FOLDER $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER
