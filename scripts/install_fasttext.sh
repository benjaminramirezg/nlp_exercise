#!/bin/bash

# This script installs FastText froim sources. It will be
# used to do language identification. A pretrained model for
# for language identification will be downloaded.

# WARNING: make sure that you have your conda environment
# activated when running this script

# To run this script you need:
#   - git
#   - wget

# Downloading and installing fasttext from source
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
cd ..
rm -rf fastText

# Downloading FastText model for language identification
cd ../models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz