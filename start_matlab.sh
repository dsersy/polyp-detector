#!/bin/bash

# Matlab directory; set only if not already set
MATLABDIR=${MATLABDIR:-/usr/local/MATLAB/R2016b}

# Get the project's root directory (i.e., the location of this script)
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Quit on error
set -e


# Caffe
CAFFE_LIB="${ROOT_DIR}/external/cnn-feature-extractor/external/caffe-bin/lib64"
export LD_LIBRARY_PATH=${CAFFE_LIB}:${LD_LIBRARY_PATH}


# Run MATLAB
${MATLABDIR}/bin/matlab -r "run ${ROOT_DIR}/startup.m"

