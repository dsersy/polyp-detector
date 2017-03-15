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
*and* the relevant celestial bodies are correctly aligned, it might work for you as well. But
it is just as (if not more) likely to fail in mysterious ways
and spew cryptic error messages instead. Or crash your Matlab (which is
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

## Running the experiments

The Matlab code can be roughly divided into the polyp detection pipeline
and the experiment framework functions, which correspond to the experiments
outlined in the MEPS paper [2].

Once you have set up the code and datasets, and started the Matlab as
outlined in the previous sections, you can perform the following steps
(the following Matlab commands are run from inside the working directory).

### Part 1a: consistency of human annotators
The first part of experimental evaluation involves estimation of consistency
of human annotators on the first seven images from the dataset-martin:
```Matlab
experiment1_evaluate_experts();
```
The above program loads the annotations made by each human annotators, and
uses the evaluation framework (the same as later used to evaluate the detector
pipeline) to compare them to the ground-truth annotations.

The results are printed in form of tab-separated table, and correspond to
values found in Table 1, Table 2, and Table S1 in the MEPS paper [2].

### Part 1b: leave-one-out evaluation of detector pipeline

Similarly, to perform the leave-one-out evaluation of the detector
pipeline on the first seven images from the dataset-martin, run:
```Matlab
experiment1_leave_one_out();
```
The whole experiment may take a while; for each of the seven images, it
takes the other six images, and uses them to train both the first stage
of the pipeline (i.e., the ACF detector) and the second stage of the
pipeline (i.e., SVM on top of CNN features extracted from proposed regions).
It then uses the trained pipeline to process the held-out image, and
evaluates the detection results.

The obtained values correspond to those listed in Table 3 in the MEPS
paper [2].

#### Visualization

To visualize the results as the test images are processed, use the
'visualize_detections' option. The experiment code also caches the
results, so before running the experiment again, either remove the old
results directory or use a different one via 'output_dir' option:
```Matlab
experiment1_leave_one_out('output_dir', 'experiment1-leave-one-out-viz', 'visualize_detections', true);
```
The example above will create Matlab .fig files in the result directory;
to load and display them, use:
```Matlab
openfig('experiment1-leave-one-out-viz/01.01-detection.fig', 'visible');
```
The figures are relatively large and may take some time to open and display.
Each figure shows masked input image, centroids of ground-truth annotations
(+) and centroids of obtained detections (x). The links between them
denote the assignments done during evaluation to determine TP, FP, and FN
(the maximum allowed distance is determined by size of annotated polyps, and
enforces reasonably-local assignments).


### Part 2: comparison on large-scale dataset

The second experiment (feasibility study for application to population
dynamics analysis) involves training the pipeline on one, relatively
small, annotated dataset, and using it to process a different, larger
dataset:
```Matlab
experiment2_train_and_evaluate();
```
The above program trains both stages of detector pipeline, and then
uses it to process 60 images from study of (Hocevar et al., 2016).

Alternatively, it is possible to run it with similarly-trained default
ACF detector that comes with the data package:
```Matlab
experiment2_train_and_evaluate('output_dir', 'experiment2-default', 'acf_detector_file', 'code/detector/acf-polyp-default.mat');
```

Again, visualization of results can be enabled with 'visualize_detections'
option.

The obtained detections are compared to the manual annotations in terms
of precision and recall. However, as the manual annotations are not
actual ground-truth annotations (they were obtained by a single person
in a single-pass annotation process), the values do not reflect the
absolute performance of the algorithm.

In MEPS paper [2] in Figures 5 and 6, the obtained detections (and
manual annotations) were used to estimate polyp densities (in smaller ROIs),
and these values are compared instead. But even comparing the detections,
the same trend can be observed (i.e., images where detector performs closer
to human annotator, and where it performs differently).


## Troubleshooting

As mentioned at the beginning of this README, it is quite likely that
running this code will crash your Matlab. This is because on errors,
the Caffe library calls abort() instead of throwing an exception, thus
bringing down the whole program (in this case, Matlab). If this happens,
see the terminal you ran the Matlab from, as the final messages should
contain the clue as to what kind of error was encountered.

Some possibilities:
- missing network-weight data file. Make sure that the detector data file
  was extracted inside the code directory. I.e., given the working
  directory polyp-detector, the following directory should exist:
  polyp-detector/code/detector/cnn-rcnn

- trying to use GPU codepath (default) on non-CUDA build. If you do not
  have CUDA installed, open file PolypDetector.m, find static method
  feature_extractor_imagenet_fc7, and change the 'use_gpu' flag in the
  feature extractor preset from true to false.

- insufficient GPU RAM: with the default parametrization of the CNN
  feature extractor, the processing requires around 1.5 GB of GPU RAM.
  This may be somewhat alleviated by reducing the batch size in feature
  extractor preset (see previous item).
