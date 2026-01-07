# this script helps the user download and set up the dataset as required for calibration etc.

# DATASET_DIR is the relative path to the HistologyHSI-GB dataset
# this file is being run from inside datasets
DATASET_DIR="../../datasets/'PKG - HistologyHSI-GB'"
RAW_DIR="./raw"
WHITE_DIR="./white"
DARK_DIR="./dark"
PREPROCESSED_DIR="./preprocessed"
RGB_DIR="./rgb"
# train and test datasets should be inside ./datasets/histology for training purposes
TRAIN_A="./histology/trainA"
TEST_A="./histology/testA"
TRAIN_B="./histology/trainB"
TEST_B="./histology/testB"
# create expected dataset folder structure for preprocessing
mkdir -p ${RAW_DIR} ${WHITE_DIR} ${DARK_DIR} ${PREPROCESSED_DIR} ${RGB_DIR} ${TRAIN_A} ${TEST_A} ${TRAIN_B} ${TEST_B}




# HistologyHSI-GB dataset is downloaded into ../../../datasets/HistologyHSI-GB

# Step 1: Organise files into subfolders
# rename files then move

# organise hdr files
# TODO: get this running - it will sort raw files but not raw.hdr
for hdr_path in ../../../datasets/raw_hdrs/raw_hdrs/P*/ROI*/raw.hdr; do
    # extract P* and ROI* parts from path
    # ROI* part includes cover image number and tumor marker  eg. ROI_1_C01_T
    patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    roi=$(basename "$(dirname "$hdr_path")")

    new_name="${patient}_${roi}_raw.hdr"

    mv "$hdr_path" "${RAW_DIR}/${new_name}"
done

# organise hsi files
# this is done
#for hdr_path in ../../../datasets/HistologyHSI-GB/P*/ROI*/raw; do
    # extract P* and ROI* parts from path
    # ROI* part includes cover image number and tumor marker  eg. ROI_1_C01_T
    #patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    #roi=$(basename "$(dirname "$hdr_path")")

    #new_name="${patient}_${roi}_raw"

    #mv "$hdr_path" "${RAW_DIR}/${new_name}"
#done

# organise whiteReference hdr files
#for hdr_path in ../../../datasets/HistologyHSI-GB/P*/ROI*/whiteReference.hdr; do
    # extract P* and ROI* parts from path
    #patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    #roi=$(basename "$(dirname "$hdr_path")")

    #new_name="${patient}_${roi}_whiteReference.hdr"

   # mv "$hdr_path" "${WHITE_DIR}/${new_name}"
#done

# organise whiteReference files
for hdr_path in ../../../datasets/HistologyHSI-GB/P*/ROI*/whiteReference; do
    extract P* and ROI* parts from path
    patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    roi=$(basename "$(dirname "$hdr_path")")

    new_name="${patient}_${roi}_whiteReference"

    mv "$hdr_path" "${WHITE_DIR}/${new_name}"
done

# organise darkReference hdr files
#for hdr_path in ../../../datasets/HistologyHSI-GB/P*/ROI*/darkReference.hdr; do
    # extract P* and ROI* parts from path
    #patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    #roi=$(basename "$(dirname "$hdr_path")")

    #new_name="${patient}_${roi}_darkReference.hdr"

    #mv "$hdr_path" "${DARK_DIR}/${new_name}"
#done

# organise darkReference hdr files
for hdr_path in ../../../datasets/HistologyHSI-GB/P*/ROI*/darkReference; do
    extract P* and ROI* parts from path
    patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    roi=$(basename "$(dirname "$hdr_path")")

    new_name="${patient}_${roi}_darkReference"

    mv "$hdr_path" "${DARK_DIR}/${new_name}"
done

# organise rgb files
#for hdr_path in ../../../datasets/HistologyHSI-GB/P*/ROI*/rgb.png; do
    # extract P* and ROI* parts from path
    #patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    #roi=$(basename "$(dirname "$hdr_path")")

    #new_name="${patient}_${roi}_rgb.hdr"

    #mv "$hdr_path" "${RGB_DIR}/${new_name}"
#done


#echo "Dataset organisation complete!"
#echo "Raw hyperspectral data:     ./datasets/raw"
#echo "White reference:      ./datasets/white"
#echo "Dark reference:       ./datasets/dark"
#echo "rgb data:     ./datasets/rgb"
#echo "Ready for preprocessing script (scripts/preprocess.py)"