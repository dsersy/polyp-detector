classdef PolypDetector_MNIST_proba < PolypDetector
    % POLYPDETECTOR_MNIST_PROBA - Polyp detector, using MNIST-trained LeNet 
    % CNN with 'proba' layer input
    methods
        function self = PolypDetector_MNIST_proba ()
            root_dir = fileparts(mfilename('fullpath'));
            
            % ACF detector
            acf_detector_name = fullfile(root_dir, 'detector', 'acf-polyp-default.mat');
            
            % Arguments for CNN feature extractor
            cnn_dir = fullfile(root_dir, 'detector', 'cnn-mnist');
            
            cnn_arguments = { ...
                fullfile(cnn_dir, 'lenet.prototxt'), ...
                fullfile(cnn_dir, 'lenet_iter_10000.caffemodel'), ...
                'layer_name', 'prob', ...
                'pixel_scale', 1/256, ...
                'use_gpu', true ...
            };

            self@PolypDetector('acf_detector', acf_detector_name, 'cnn_arguments', cnn_arguments);
        end
    end
end