#!/usr/bin/env bash

### EDIT THIS TO WHEREVER YOU'RE STORING YOU DATA ###
# folder should exist before you mount it
BASE_DIR=/home/abersier@acfr.usyd.edu.au/Thesis/ros2_ws/dyno_pipeline

LOCAL_DATA_FOLDER=$BASE_DIR/data/datasets/
LOCAL_RESULTS_FOLDER=$BASE_DIR/data/results/
LOCAL_DYNO_SAM_FOLDER=$BASE_DIR/code/core/DynOSAM
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=$BASE_DIR/code/extras/

bash create_container_base.sh acfr_rpg/dyno_sam_cuda dyno_sam $LOCAL_DATA_FOLDER $LOCAL_RESULTS_FOLDER $LOCAL_DYNO_SAM_FOLDER $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER
