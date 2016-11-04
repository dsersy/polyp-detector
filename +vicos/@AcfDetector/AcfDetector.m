classdef AcfDetector < handle
    % ACFDETECTOR - ACF detector wrapper
    %
    % (C) 2016, Rok Mandeljc <rok.mandeljc@fri.uni-lj.si>
    
    properties
        % Detector name (= basename of detector file)
        name 
        
        % Detector
        detector
        
        % Original NMS settings
        nms
    end
    
    %% Public API
    methods
        function self = AcfDetector (detector_file)
            % self = ACFDETECTOR (detector_file)
            %
            % Creates an ACF detector from the given detector file. This
            % function stores the detector's NMS settings, and disables the
            % NMS in the detector itself.
            %
            % Input:
            %  - detector_file: ACF detector file
            %
            % Output:
            %  - self: @AcfDetector instance
            
            % Detector basename
            [ ~, self.name ] = fileparts(detector_file);
            
            % Load detector
            tmp = load(detector_file);
            self.detector = tmp.detector;
            
            % Copy NMS settings
            self.nms = self.detector.opts.pNms;
            self.nms.maxn = 5000; % Split boxes in batches of 5000 to improve NMS performance
            
            % Disable NMS on detector
            self.detector.opts.pNms.type = 'none';
        end
    end
    
    methods
        % Detection
        [ boxes, all_boxes, time_det, time_nms ] = detect (self, I, varargin)

        % Image upscaling (on ACF detector level)
        set_upscale_image (self, factor)
        factor = get_upscale_image (self)
    end
    
    %% Detector training
    methods (Static)
        training_prepare_dataset (training_images, output_path, varargin)
        detector = training_train_detector (data_dir, varargin)
    end
end