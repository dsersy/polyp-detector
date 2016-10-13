#!/bin/sh

# Matlab
MATLABDIR=${MATLABDIR:-/usr/local/MATLAB/R2015b}

# Caffe
CAFFE_LIB=$(pwd)/external/cnn-feature-extractor/external/caffe-bin/lib/

# Set path
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CAFFE_LIB}

# Run MATLAB
${MATLABDIR}/bin/matlab

