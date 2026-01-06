# this script helps the user download and set up the dataset as required for calibration etc.

# DATASET_DIR is the relative path to the HistologyHSI-GB dataset
DATASET_DIR="../../datasets/'PKG - HistologyHSI-GB'"
RAW_DIR="./datasets/raw"
WHITE_DIR="./datasets/white"
DARK_DIR="./datasets/dark"
PREPROCESSED_DIR="./datasets/preprocessed"
RGB_DIR="./datasets/rgb"
# train and test datasets should be inside ./datasets/histology for training purposes
TRAIN_A="./datasets/histology/trainA"
TEST_A="./datasets/histology/testA"
TRAIN_B="./datasets/histology/trainB"
TEST_B="./datasets/histology/testB"
# create expected dataset folder structure for preprocessing
mkdir -p ${RAW_DIR} ${WHITE_DIR} ${DARK_DIR} ${PREPROCESSED_DIR} ${RGB_DIR} ${TRAIN_A} ${TEST_A} ${TRAIN_B} ${TEST_B}




# the dataset should have already been downloaded into /datasets/temp

# Step 1: Organise files into subfolders
# rename files then move

# organise hdr files
for hdr_path in ../../datasets/HistologyHSI-GB/P*/ROI*/raw.hdr; do
    # extract P* and ROI* parts from path
    # ROI* part includes cover image number and tumor marker  eg. ROI_1_C01_T
    patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    roi=$(basename "$(dirname "$hdr_path")")

    new_name="${patient}_${roi}_raw.hdr"

    mv "$hdr_path" "${RAW_DIR}/${new_name}"
done

# organise hsi files
for hdr_path in "${DATASET_DIR}"/temp/P*/ROI*/raw; do
    # extract P* and ROI* parts from path
    # ROI* part includes cover image number and tumor marker  eg. ROI_1_C01_T
    patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    roi=$(basename "$(dirname "$hdr_path")")

    new_name="${patient}_${roi}_raw"

    mv "$hdr_path" "${RAW_DIR}/${new_name}"
done

# organise whiteReference files
for hdr_path in "${DATASET_DIR}"/temp/P*/ROI*/whiteReference.hdr; do
    # extract P* and ROI* parts from path
    patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    roi=$(basename "$(dirname "$hdr_path")")

    new_name="${patient}_${roi}_whiteReference.hdr"

    mv "$hdr_path" "${WHITE_DIR}/${new_name}"
done

# organise darkReference files
for hdr_path in "${DATASET_DIR}"/temp/P*/ROI*/darkReference.hdr; do
    # extract P* and ROI* parts from path
    patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    roi=$(basename "$(dirname "$hdr_path")")

    new_name="${patient}_${roi}_darkReference.hdr"

    mv "$hdr_path" "${DARK_DIR}/${new_name}"
done

# organise rgb files
for hdr_path in "${DATASET_DIR}"/temp/P*/ROI*/rgb.png; do
    # extract P* and ROI* parts from path
    patient=$(basename "$(dirname "$(dirname "$hdr_path")")")
    roi=$(basename "$(dirname "$hdr_path")")

    new_name="${patient}_${roi}_rgb.hdr"

    mv "$hdr_path" "${RGB_DIR}/${new_name}"
done


echo "Dataset organisation complete!"
echo "Raw hyperspectral data:     ./datasets/raw"
echo "White reference:      ./datasets/white"
echo "Dark reference:       ./datasets/dark"
echo "rgb data:     ./datasets/rgb"
echo "Ready for preprocessing script (scripts/preprocess.py)"