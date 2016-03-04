classdef PolypDetector_MNIST_fc1 < PolypDetector
    % POLYPDETECTOR_MNIST_FC1 - Polyp detector, using MNIST-trained LeNet 
    % CNN with 'ip1' (first fully-connected) layer input
    methods
        function self = PolypDetector_MNIST_fc1 ()
            root_dir = fileparts(mfilename('fullpath'));

            % Cache folder
            cache_dir = 'cache-mnist-fc1';
            
            % ACF detector
            acf_detector_name = fullfile(root_dir, 'detector', 'acf-polyp-default.mat');
            
            % Arguments for CNN feature extractor
            cnn_dir = fullfile(root_dir, 'detector', 'cnn-mnist');
            
            cnn_arguments = { ...
                fullfile(cnn_dir, 'lenet.prototxt'), ...
                fullfile(cnn_dir, lenet_iter_10000.caffemodel'), ...
                'layer_name', 'ip1', ...
                'pixel_scale', 1/256, ...
                'use_gpu', true ...
            };

            self@PolypDetector('cache_dir', cache_dir, 'acf_detector', acf_detector_name, 'cnn_arguments', cnn_arguments);
        end
    end
end