#!/bin/bash

# Matlab directory; set only if not already set
MATLABDIR=${MATLABDIR:-/usr/local/MATLAB/R2016b}

# Get the project's root directory (i.e., the location of this script)
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Quit on error
set -e

########################################################################
#                     Build CNN Feature Extractor                      #
########################################################################
echo "Building CNN Feature extractor dependencies..."
"${ROOT_DIR}/external/cnn-feature-extractor/build_all.sh"

########################################################################
#                           Build LIBLINEAR                            #
########################################################################
echo "Building LIBLINEAR..."
make MATLABDIR="$MATLABDIR" -C "${ROOT_DIR}/external/liblinear/matlab"
