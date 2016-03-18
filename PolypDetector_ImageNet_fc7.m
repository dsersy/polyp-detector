classdef PolypDetector_ImageNet_fc7 < PolypDetector
    % POLYPDETECTOR_IMAGENET_FC7 - Polyp detector, using ImageNet-trained
    % AlexNet CNN with 'fc7' (second fully-connected) layer input
    methods
        function self = PolypDetector_ImageNet_fc7 ()
            root_dir = fileparts(mfilename('fullpath'));
            
            % ACF detector
            acf_detector_name = fullfile(root_dir, 'detector', 'acf-polyp-default.mat');
            
            % Arguments for CNN feature extractor
            cnn_dir = fullfile(root_dir, 'detector', 'cnn-rcnn');
            
            cnn_arguments = { ...
                fullfile(cnn_dir, 'rcnn_batch_256_output_fc7.prototxt'), ...
                fullfile(cnn_dir, 'finetune_voc_2012_train_iter_70k'), ...
                'layer_name', 'fc7', ...
                'padding', 16, ...
                'square_crop', false, ...
                'pixel_means', fullfile(cnn_dir, 'pixel_means.mat'), ...
                'use_gpu', true ...
            };
        
            self@PolypDetector('acf_detector', acf_detector_name, 'cnn_arguments', cnn_arguments);
        end
    end
end