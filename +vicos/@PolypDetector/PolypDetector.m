classdef PolypDetector < handle
    % POLYPDETECTOR - Polyp detector class
    %
    % This class implements the full polyp detector pipeline, consisting of
    % pre-trained ACF detector for region proposals, CNN feature
    % extraction, and SVM classification.
    %
    % (C) 2016 Rok Mandeljc <rok.mandeljc@fri.uni-lj.si>
    
    properties
        %% Detection pipeline components
        acf_detector
        cnn_extractor
        svm_classifier
        
        % Function handles for creating the above components (so that we
        % actually create them on-demand)
        acf_detector_factory
        feature_extractor_factory
        classifier_factory
        
        %% Pipeline parameters
        % Non-maxima suppresion overlap threshold for ACF detection step
        acf_nms_overlap = 0.5
        
        % Additional scaling applied to the ACF detections.
        acf_box_scaling = 1.0
        
        % Positive and negative overlap for SVM samples. An ACF detection 
        % is taken as a positive sample if overlap is greater or equal to
        % training_positive_overlap, and negative if overlap is smaller
        % than training_negative_overlap. The boxes with overlap that fall
        % between the values are discarded from SVM training (applicable
        % only if traning_positive_overlap and training_negative_overlap
        % have different values)
        training_positive_overlap = 0.1
        training_negative_overlap = 0.05
        
        % L2-normalize feature vectors
        l2_normalized_features = true
        
        % Non-maxima suppression overlap threshold for confirmed detections
        svm_nms_overlap = 0.1
        
        % Evaluation overlap threshold
        evaluation_overlap = 0.1
        
        % Evaluation distance threshold (in pixels)
        evaluation_distance = 32
    end
        
    %% Public API
    methods
        function self = PolypDetector (varargin)
            % self = POLYPDETECTOR (varargin)
            %
            % Creates a new instance of @PolypDetector.
            %
            % Input: key/value pairs
            %  - region_proposal_factory: handle to function that creates
            %    an instance of region proposal generator. By default, a 
            %    pre-trained ACF detector is used.
            %  - feature_extractor_factory: handle to function that creates
            %    an instance of feature extractor. By default, CNN feature
            %    extractor using ImageNet FC7 features is used.
            %  - classifier_factory: handle to function that creates a
            %    classifier. By default, a LIBLINEAR binary SVM is used.
            %
            % Output:
            %  - self:
            
            % Input parser
            parser = inputParser();
            parser.addParameter('acf_detector_factory', @() vicos.AcfDetector(fullfile(self.get_root_code_path(), 'detector', 'acf-polyp-default.mat')));
            parser.addParameter('feature_extractor_factory', @() self.feature_extractor_imagenet_fc7());
            parser.addParameter('classifier_factory', @() vicos.svm.LibLinear());
            parser.parse(varargin{:});
            
            self.acf_detector_factory = parser.Results.acf_detector_factory;
            assert(isa(self.acf_detector_factory, 'function_handle') || isempty(self.acf_detector_factory), 'acf_detector_factory must be a function handle!');
            
            self.feature_extractor_factory = parser.Results.feature_extractor_factory;
            assert(isa(self.feature_extractor_factory, 'function_handle') || isempty(self.feature_extractor_factory), 'feature_extractor_factory must be a function handle!');
            
            self.classifier_factory = parser.Results.classifier_factory;
            assert(isa(self.classifier_factory, 'function_handle') || isempty(self.classifier_factory), 'classifier_factory must be a function handle!');
        end
    end
        
    
    methods
        % Processing pipeline steps (generally not meant to be used outside
        % this class)
        [ regions, time_det, time_nms ] = detect_candidate_regions (self, I, cache_file)
        [ features, time ] = extract_features_from_regions (self, I, regions, cache_file)
    
        % Batch train/evaluation methods
        train_and_evaluate (self, result_dir, varargin)
        leave_one_out_cross_validation (self, result_dir, varargin)
        
        % Detector/classifier loading
        load_acf_detector (self, filename)
        load_classifier (self, filename)
        
        % Main image processing function 
        detections = process_image (self, image_filename, varargin)
        
        % Classifier training
        svm = train_svm_classifier (self, varargin)
        
        % Utility functions
        identifier = construct_classifier_identifier (self)
    end
    
    methods (Static)
        function root_dir = get_root_code_path ()
            % root_dir = GET_ROOT_CODE_PATH ()
            %
            % Retrieves the project's root code directory.
            %
            % Output:
            %  - root_dir: project's root code directory
            
            % Path to this .m file
            root_dir = fileparts(mfilename('fullpath'));
            
            % Move out to the root code directory
            root_dir = fullfile(root_dir, '..', '..');
        end
        
        % Data loading
        [ I, basename, poly, boxes, manual_annotations ] = load_data (image_filename)
        
        % Box enlargement
        boxes = enlarge_boxes (boxes, scale_factor)

        % Evaluation
        [ gt, dt ] = evaluate_detections (detections, annotations, varargin)
        [ gt, dt ] = evaluate_detections_as_points (detections, annotations, varargin)

        % Visualization
        fig = visualize_detections_as_boxes (I, polygon, annotations, detections, varargin)
        fig = visualize_detections_as_points (I, polygon, annotations, detections, varargin)
    end
    
    
    %% Feature extractor presets
    methods (Static)
        function extractor = feature_extractor_imagenet_fc7 ()
            % extractor = FEATURE_EXTRACTOR_IMAGENET_FC7 ()
            %
            % Creates a ImageNet FC7 CNN feature extractor.
            
            % Arguments for CNN feature extractor
            cnn_dir = fullfile(vicos.PolypDetector.get_root_code_path(), 'detector', 'cnn-rcnn');
            
            cnn_arguments = { ...
                fullfile(cnn_dir, 'rcnn_batch_256_output_fc7.prototxt'), ...
                fullfile(cnn_dir, 'finetune_voc_2012_train_iter_70k'), ...
                'layer_name', 'fc7', ...
                'padding', 16, ...
                'square_crop', false, ...
                'pixel_means', fullfile(cnn_dir, 'pixel_means.mat'), ...
                'use_gpu', true ...
            };
            
            % Create CNN feature extractor
            extractor = CnnFeatureExtractor(cnn_arguments{:});
        end
    end
    
    %% Default test and train images
    properties
        default_train_images = { ...
            'dataset-kristjan/07.03.jpg', ...
            'dataset-kristjan/13.01.jpg', ...
            'dataset-kristjan/13.03.jpg', ...
            'dataset-kristjan/13.04.jpg', ...
            'dataset-kristjan/13.05.jpg' };
        
        default_test_images = {
            'dataset-kristjan/01.01.jpg', ...
            'dataset-kristjan/01.02.jpg', ...
            'dataset-kristjan/01.03.jpg', ...
            'dataset-kristjan/01.04.jpg', ...
            'dataset-kristjan/01.05.jpg', ...
            'dataset-kristjan/02.01.jpg', ...
            'dataset-kristjan/02.02.jpg', ...
            'dataset-kristjan/02.03.jpg', ...
            'dataset-kristjan/02.04.jpg', ...
            'dataset-kristjan/02.05.jpg', ...
            'dataset-kristjan/03.01.jpg', ...
            'dataset-kristjan/03.02.jpg', ...
            'dataset-kristjan/03.03.jpg', ...
            'dataset-kristjan/03.04.jpg', ...
            'dataset-kristjan/03.05.jpg', ...
            'dataset-kristjan/04.01.jpg', ...
            'dataset-kristjan/04.02.jpg', ...
            'dataset-kristjan/04.03.jpg', ...
            'dataset-kristjan/04.04.jpg', ...
            'dataset-kristjan/04.05.jpg', ...
            'dataset-kristjan/05.01.jpg', ...
            'dataset-kristjan/05.02.jpg', ...
            'dataset-kristjan/05.03.jpg', ...
            'dataset-kristjan/06.01.jpg', ...
            'dataset-kristjan/06.02.jpg', ...
            'dataset-kristjan/06.03.jpg', ...
            'dataset-kristjan/07.01.jpg', ...
            'dataset-kristjan/07.02.jpg', ...
            'dataset-kristjan/08.01.jpg', ...
            'dataset-kristjan/08.03.jpg' };
    end
end