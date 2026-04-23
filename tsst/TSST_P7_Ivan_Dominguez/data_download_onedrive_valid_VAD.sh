#!/bin/bash

# INSTRUCTIONS:
# Run this script from the terminal:  ./data_download_onedrive_valid_VAD.sh . Files will be downloaded in ./data folder

mkdir ./data

wget https://dauam-my.sharepoint.com/:u:/g/personal/alicia_lozano_uam_es/ESc5XzkpZ3ZBnGFQ6HWdn_UB38NHMOLTLtTcEE_b81Cylw?download=1 
mv ESc5XzkpZ3ZBnGFQ6HWdn_UB38NHMOLTLtTcEE_b81Cylw?download=1 ./data/valid_VAD.zip
unzip ./data/valid_VAD.zip > /dev/null
mv ./valid_VAD ./data/.



