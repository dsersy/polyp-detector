# PoCo: polyp* detector/counter #
(* the jellyfish kind (spychistocoma), not the medical kind)

**Prototype & evaluation framework**

(C) 2016-2017, Rok Mandeljc


## Summary

This project contains the prototype implementation of the PoCo polyp
detector/counter, presented in:

1. M. Vodopivec et al., Polyp counting made easy: two stage scyphistoma
    detection for a computer-assisted census in underwater imagery,
    Fifth International Jellyfish Bloom Symposium: Abstract book,
    Barcelona, 2016

2. M. Vodopivec et al., Polyp counting made easy: towards automated
    scyphistoma census in underwater imagery, submitted to MEPS

The code is provided as supplement to the MEPS submission [2] in case
an interested party wishes to reproduce the experimental results from
the paper. The accompanying datasets are available here:

http://vision.fe.uni-lj.si/~rokm/polyp-detector

The sections below outline the installation and setup, and steps needed
to reproduce the experiments.


## Disclaimer

This is an **academic-grade software prototype**, provided as-is
and without support beyond the included documentation. It works for me on my computers, and
perhaps, if your karma is good enough, the deity of your choice is having a good day,
*and* the relevant celestial bodies are correctly aligned, it might work for you too. But
it is just as (if not more) likely to fail in mysterious ways
and spew cryptic error messages. Or crash your Matlab (which is
actually quite likely due to the finicky nature of error
handling in the Caffe library). There, you cannot say you were not warned...


## Prerequisites

Recent 64-bit linux distribution (preferably Fedora or Ubuntu),
recent Matlab with Image processing toolbox and working MEX compiler.
Additional required packages are listed below.

For reference, the code was primarily developed and tested on a mid-range
workstation (Quad-core Intel i5-3570K, 16 GB RAM, NVIDIA GeForce GTX 970
GPU with 4 GB RAM), running 64-bit Fedora 24/25, with Matlab
R2016a/R2016b, CUDA 8 and CuDNN 5.1.

The following packages are required for building Caffe, which is used for
CNN feature extraction. For Fedora (assuming appripriate repositories
are enabled):

```Shell
sudo dnf install cmake gcc-c++ boost-devel glog-devel gflags-devel \
    protobuf-devel hdf5-devel lmdb-devel leveldb-devel snappy-devel \
    openblas-devel python-devel python2-numpy
sudo dnf install  cuda-devel cudnn-devel
```
and for Ubuntu:
```Shell
sudo apt-get install cmake build-essential libboost-all-dev libgflags-dev \
    libgoogle-glog-dev libprotobuf-dev protobuf-compiler libhdf5-dev \
    liblmdb-dev libleveldb-dev libsnappy-dev libopenblas-dev python-dev \
    python-numpy
sudo apt-get install nvidia-cuda-toolkit
```
The CUDA depndency is optional, however, the code currently assumes
a CUDA-enabled Caffe build. If you build Caffe without CUDA (by not
having it installed when executing the steps outlined below), you will
likely need to change 'use_gpu' flag in the feature extractor preset
(file PolypDetector.m, static method feature_extractor_imagenet_fc7).

## Installation

Fire up your favorite shell, and export path to your Matlab installation:
```Shell
export MATLABDIR=/usr/local/MATLAB/R2016b
```

Create a working directory. Unless stated otherwise, the rest of
instructions assumes that commands are run from this directory (both shell and Matlab):
```Shell
mkdir polyp-detector
cd polyp-detector
```

Download and unpack the datasets:
```Shell
wget http://vision.fe.uni-lj.si/~rokm/polyp-detector/dataset-kristjan.tar.xz
tar xJf dataset-kristjan.tar.xz

wget http://vision.fe.uni-lj.si/~rokm/polyp-detector/dataset-martin.tar.xz
tar xJf dataset-martin.tar.xz

wget http://vision.fe.uni-lj.si/~rokm/polyp-detector/dataset-sara.tar.xz
tar xJf dataset-sara.tar.xz

rm -fr dataset-kristjan.tar.xz dataset-martin.tar.xz dataset-sara.tar.xz
```

Checkout the code from git repository:
```Shell
git clone https://github.com/rokm/polyp-detector code
cd code
git submodule update --init --recursive
cd ..
```

Download and unpack the default detector data (the archived "detector"
directory must be extracted inside the code directory):
```Shell
cd code
wget http://vision.fe.uni-lj.si/~rokm/polyp-detector/detector-data.tar.xz
tar xJf detector-data.tar.xz
rm -fr detector-data.tar.xz
cd ..
```

Build all dependencies:
```Shell
./code/build_all.sh
```

The above script will attempt to build the version of Caffe that is
bundled with the cnn-feature-extractor module, and LIBLINEAR for Matlab,
and ensures that the resulting binaries are places in the expected
location within the code directory.

If all went well, the build script should finish without errors. Otherwise,
check the error message to see what the issue is (incorrectly checked-out source,
missing dependencies, etc.).


## Running Matlab

On linux, the external libraries (i.e., the Caffe shared library) need to
be in LD_LIBRARY_PATH before Matlab is started. To simplify the launch
process, we provide a wrapper script that sets the proper library paths,
starts Matlab, and executes startup.m script to include external Matlab-side
dependencies.

Therefore, to start Matlab, move to the working directory, and make
sure to export path to your Matlab installation:
```Shell
cd polyp-detector
export MATLABDIR=/usr/local/MATLAB/R2016b
```

Then simply run the wrapper script:
```Shell
./code/start_matlab.sh
```
