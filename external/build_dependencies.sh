#!/bin/bash

MATLABDIR=${MATLABDIR:-/usr/local/MATLAB/R2015b}

EXTERNAL_ROOT=$(pwd)


# Quit on error
set -e


########################################################################
#                     Build CNN Feature Extractor                      #
########################################################################
echo "Building CNN Feature extractor dependencies..."
pushd cnn-feature-extractor/external

./build_dependencies.sh

popd


########################################################################
#                           Build LIBLINEAR                            #
########################################################################
echo "Building LIBLINEAR..."
pushd liblinear/matlab
make MATLABDIR=$MATLABDIR
popd
