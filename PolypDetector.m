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
        training_positive_overlap = 0.5
        training_negative_overlap = 0.1
        
        % Function handle for creating new SVM classifier
        svm_create_function = @() classifier.LIBLINEAR()
        
        % Non-maxima suppression overlap threshold for confirmed detections
        svm_nms_overlap = 0.1
        
        % Evaluation overlap
        evaluation_overlap = 0.1
    end
    
    properties (Access = private)
        cnn_arguments
    end
        
    %% Public API
    methods
        function self = PolypDetector (varargin)
            % self = POLYPDETECTOR (varargin)
            %
            % Creates a new instance of @PolypDetector.
            %
            % Input: key/value pairs
            %  - acf_detector: ACF detector .mat filename
            %  - cnn_arguments: cell array of key/value arguments to pass
            %    to @CnnFeatureExtractor
            %
            % Output:
            %  - self: @PolypDetector instance
            
            % Input parser
            parser = inputParser();
            parser.addParameter('acf_detector', 'detector/acf-polyp-default.mat', @ischar);
            parser.addParameter('cnn_arguments', {}, @iscell);
            parser.parse(varargin{:});
            
            % Create ACF detector wrapper
            acf_detector_file = parser.Results.acf_detector;
            self.acf_detector = AcfDetector(acf_detector_file);
            
            % Create CNN feature extractor
            self.cnn_arguments = parser.Results.cnn_arguments;
            self.cnn_extractor = CnnFeatureExtractor(self.cnn_arguments{:});
        end
    end
        
    % Processing pipeline steps (generally not meant to be used outside
    % this class)
    methods
        function [ I, basename, poly, boxes, manual_annotations ] = load_data (self, image_filename)
            % [ I, basename, poly, boxes, manual_annotations ] = LOAD_DATA (self, image_filename)
            %
            % Loads an image and its accompanying polygon and bounding box
            % annotations, if available.
            %
            % Input:
            %  - self: @PolypDetector instance
            %  - image_filename: input image filename
            %
            % Output:
            %  - I: loaded image
            %  - basename: image's basename, which can be passed to
            %    subsequent processing functions for data caching
            %  - poly: polygon that describes ROI
            %  - boxes: manually-annotated bounding boxes
            
            % Get path and basename
            [ pathname, basename, ~ ] = fileparts(image_filename);
            
            % Load image
            I = imread(image_filename);
            
            % Load polygon
            if nargout > 2,
                poly_file = fullfile(pathname, [ basename, '.poly' ]);
                if exist(poly_file, 'file'),
                    poly = load(poly_file);
                else
                    poly = [];
                end
            end
            
            % Load annotations (boxes)
            if nargout > 3,
                boxes_file = fullfile(pathname, [ basename, '.bbox' ]);
                if exist(boxes_file, 'file'),
                    boxes = load(boxes_file);
                else
                    boxes = [];
                end
            end
            
            % Load manual annotations (points)
            if nargout > 4,
                annotation_files = dir( fullfile(pathname, [ basename, '.manual-*.txt' ]) );
                
                manual_annotations = cell(numel(annotation_files), 2);
                
                for f = 1:numel(annotation_files),
                    % Annotation file
                    annotation_file = fullfile(pathname, annotation_files(f).name);
                    
                    % Get basename and deduce annotation ID
                    [ ~, annotation_basename ] = fileparts(annotation_file);
                    pattern = '.manual-';
                    idx = strfind(annotation_basename, pattern) + numel(pattern);
                    
                    manual_annotations{f, 1} = annotation_basename(idx:end); % Annotation ID
                    manual_annotations{f, 2} = load(annotation_file); % Point list
                end
            end
        end
        
        function [ regions, time_det, time_nms ] = detect_candidate_regions (self, I, cache_file)
            % [ regions, time_det, time_nms ] = DETECT_CANDIDATE_REGIONS (self, I, cache_file)
            %
            % Detects candidate regions in the given image.
            %
            % Input:
            %  - self: @PolypDetector instance
            %  - I: image
            %  - cache_file: optional cache file to use (default: '')
            %
            % Output:
            %  - regions: detected regions
            %  - time_det: time spent in region detection
            %  - time_nms: time spent in the first non-maxima suppression
            %    pass
            
            if ~exist('cache_file', 'var'),
                cache_file = '';
            end
            
            if ~isempty(cache_file) && exist(cache_file, 'file'),
                % Load from cache
                tmp = load(cache_file);
                
                % Validate cache file
                assert(self.acf_nms_overlap == tmp.nms_overlap, 'Invalid cache file; non-maxima suppression overlap threshold mismatch!');
                
                % Copy from cache
                regions = tmp.regions;
                time_det = tmp.time_det;
                time_nms = tmp.time_nms;
            else
                % Run ACF detector
                [ regions, regions_all, time_det, time_nms ] = self.acf_detector.detect(I, 'nms_overlap', self.acf_nms_overlap);
                
                % Save to cache
                if ~isempty(cache_file),
                    ensure_path_exists(cache_file);
                    
                    tmp.nms_overlap = self.acf_nms_overlap;
                    tmp.regions = regions;
                    tmp.regions_all = regions_all;
                    tmp.time_det = time_det;
                    tmp.time_nms = time_nms;
                    
                    save(cache_file, '-struct', 'tmp');
                end
            end
            
            % Rescale boxes, if necessary
            if self.acf_box_scaling ~= 1.0,
                regions = enlarge_boxes(regions, self.acf_box_scaling);
            end
        end
        
        function [ features, time ] = extract_features_from_regions (self, I, regions, cache_file)
            % features = EXTRACT_FEATURES_FROM_REGIONS (self, I, regions)
            %
            % Extract CNN features from given regions.
            %
            % Input:
            %  - self: @PolypDetector instance
            %  - I: image
            %  - regions: Nx4 matrix describing regions
            %  - cache_file: optional cache file to use (default: '')
            %
            % Output:
            %  - features: DxN matrix of extracted features
            %  - time: time spent in feature extraction
            
            if ~exist('cache_file', 'var'),
                cache_file = '';
            end
            
            if ~isempty(cache_file) && exist(cache_file, 'file'),
                % Load from cache
                tmp = load(cache_file);
                
                % Validate cache file
                assert(size(regions, 1) == size(tmp.features, 2), 'Invalid cache file; mismatch between number of regions and stored feature vectors!');
                
                % Copy from cache
                features = tmp.features;
                time = tmp.time;
            else
                % Convert [ x, y, w, h ] to [ x1, y1, x2, y2 ], in 4xN format
                boxes = [ regions(:,1), regions(:,2), regions(:,1)+regions(:,3)+1, regions(:,2)+regions(:,4)+1 ]';
            
                % Extract CNN features
                [ features, time ] = self.cnn_extractor.extract(I, 'regions', boxes);
                
                % Save to cache
                if ~isempty(cache_file),
                    ensure_path_exists(cache_file);
                    
                    tmp.features = features;
                    tmp.time = time;
                    save(cache_file, '-struct', 'tmp');
                end
            end
        end
    end

    methods
        function train_and_evaluate (self, result_dir, varargin)
            % TRAIN_AND_EVALUATE (self, result_dir)
            parser = inputParser();
            parser.addParameter('train_images', self.default_train_images, @iscell);
            parser.addParameter('test_images', self.default_test_images, @iscell);
            parser.addParameter('display_svm_samples', false, @islogical);
            parser.parse(varargin{:});
            
            train_images = parser.Results.train_images;
            test_images = parser.Results.test_images;
            display_svm_samples = parser.Results.display_svm_samples;
            
            % Cache 
            cache_dir = fullfile(result_dir, 'cache');
            
            % Store old classifier
            old_classifier = self.svm_classifier;
            
            %% Train SVM
            classifier_file = fullfile(result_dir, 'classifier.mat');
            if exist(classifier_file, 'file'),
                % Load from file
                tmp = load(classifier_file);
                self.svm_classifier = tmp.classifier;
            else
                % Train
                t = tic();
                self.train_svm_classifier('train_images', train_images, 'cache_dir', cache_dir, 'display_svm_samples', display_svm_samples);
                time = toc(t);
                    
                % Save
                ensure_path_exists(classifier_file);
                
                tmp.classifier = self.svm_classifier;
                tmp.time = time;
                save(classifier_file, '-struct', 'tmp');
            end
            
            %% Test SVM
            all_results = repmat(struct(...
                                  'image_name', '', ...
                                  'tp', 0, ...
                                  'fn', 0, ...
                                  'fp', 0, ...
                                  'precision', 0, ...
                                  'recall', 0, ...
                                  'num_annotated', 0, ...
                                  'num_detected', 0), 1, 0);
                              
            for i = 1:numel(test_images),
                test_image = test_images{i};
                
                fprintf('Processing test image #%d/%d: %s\n', i, numel(test_images), test_image);
                
                %% Load test image
                [ I, basename, poly, annotations ] = self.load_data(test_image);

                % Try loading result from cache
                results_file = fullfile(result_dir, [ basename, '.mat' ]);
                if exist(results_file, 'file'),
                    results = load(results_file);
                    all_results = [ all_results, results ];
                    continue;
                end
                
                % Process
                detections = self.process_image(test_image, 'cache_dir', cache_dir);
                
                %% Evaluate
                % Create mask
                mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
            
                % Evaluate detections
                [ gt, det ] = evaluate_detections(detections, annotations, 'threshold', self.evaluation_overlap, 'multiple', false, 'validity_mask', mask);
                
                % Count
                tp = sum( gt(:,5) == 1 );
                fn = sum( gt(:,5) == 0 );
                %tp = sum( det(:,6) == 1 );
                fp = sum( det(:,6) == 0 );
                
                precision = 100*tp/(tp+fp);
                recall = 100*tp/(tp+fn);
                
                num_annotated = sum(gt(:,5) ~= -1);
                num_detected = sum(det(:,6) ~= -1);                
                
                %% Store results
                results.image_name = test_image;
                results.tp = tp;
                results.fn = fn;
                results.fp = fp;
                results.precision = precision;
                results.recall = recall;
                results.num_annotated = num_annotated;
                results.num_detected = num_detected;
                
                ensure_path_exists(results_file);
                save(results_file, '-struct', 'results');
                
                all_results(i) = results;
            end
            
            save(fullfile(result_dir, 'all_results.mat'), 'all_results');
            
            % Display results
            fprintf('\n\n');
            fprintf('IMAGE_NAME\tREC\tPREC\tRELATIVE\n');
            for i = 1:numel(all_results),
                fprintf('%s\t%3.2f\t%3.2f\t%3.2f\n', all_results(i).image_name, all_results(i).recall, all_results(i).precision, 100*all_results(i).num_detected/all_results(i).num_annotated);
            end
            fprintf('\n');
            fprintf('%s\t%3.2f\t%3.2f\t%3.2f\n', 'AVERAGE', mean([all_results.recall]), mean([all_results.precision]), 100*mean([all_results.num_detected]./[all_results.num_annotated]));
            fprintf('\n');
            
            % Restore old classifier
            self.svm_classifier = old_classifier;
        end
        
        function leave_one_out_cross_validation (self, result_dir)
            % LEAVE_ONE_OUT_CROSS_VALIDATION (self, result_dir)
                        
            % Cache 
            cache_dir = fullfile(result_dir, 'cache');
            
            % Create list of images
            images = union(self.default_train_images, self.default_test_images);
            
            % Store old classifier
            old_classifier = self.svm_classifier;
            
            %% Leave-one-out loop
            all_results = repmat(struct(...
                                  'image_name', '', ...
                                  'tp', 0, ...
                                  'fn', 0, ...
                                  'fp', 0, ...
                                  'precision', 0, ...
                                  'recall', 0, ...
                                  'num_annotated', 0, ...
                                  'num_detected', 0), 1, numel(images));
                              
            for i = 1:numel(images),
                train_images = images;
                train_images(i) = [];
                
                test_image = images{i};
                
                % Load test image
                [ I, basename, poly, annotations ] = self.load_data(test_image);

                % Try loading result from cache
                results_file = fullfile(result_dir, [ basename, '.mat' ]);
                if exist(results_file, 'file'),
                    results = load(results_file);
                    all_results(i) = results;
                    continue;
                end
                
                %% Train SVM
                classifier_file = fullfile(result_dir, 'classifiers', [ basename, '.mat' ]);
                if exist(classifier_file, 'file'),
                    % Load from file
                    tmp = load(classifier_file);
                    self.svm_classifier = tmp.classifier;
                else
                    % Train
                    t = tic();
                    self.train_svm_classifier('train_images', train_images, 'cache_dir', cache_dir);
                    time = toc(t);
                    
                    % Save
                    classifier = self.svm_classifier;
                    ensure_path_exists(classifier_file);
                    save(classifier_file, 'classifier', 'time');
                end
                
                %% Process the left-out image
                detections = self.process_image(test_image, 'cache_dir', cache_dir);
                
                %% Evaluate
                % Create mask
                mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
            
                % Evaluate detections
                [ gt, det ] = evaluate_detections(detections, annotations, 'threshold', self.evaluation_overlap, 'multiple', false, 'validity_mask', mask);
                
                % Count
                tp = sum( gt(:,5) == 1 );
                fn = sum( gt(:,5) == 0 );
                %tp = sum( det(:,6) == 1 );
                fp = sum( det(:,6) == 0 );
                
                precision = 100*tp/(tp+fp);
                recall = 100*tp/(tp+fn);
                
                num_annotated = sum(gt(:,5) ~= -1);
                num_detected = sum(det(:,6) ~= -1);                
                
                %% Store results
                results.image_name = test_image;
                results.tp = tp;
                results.fn = fn;
                results.fp = fp;
                results.precision = precision;
                results.recall = recall;
                results.num_annotated = num_annotated;
                results.num_detected = num_detected;
                
                ensure_path_exists(results_file);
                save(results_file, '-struct', 'results');
                
                all_results(i) = results;
            end
            
            save(fullfile(result_dir, 'all_results.mat'), 'all_results');
            
            % Restore old classifier
            self.svm_classifier = old_classifier;
        end
    end
    
    
    methods
        function load_acf_detector (self, filename)
            % LOAD_ACF_DETECTOR (self, filename)
            %
            % Loads an ACF detector from file
            
            if ~exist('filename', 'var') || isempty(filename),
                [ filename, pathname ] = uigetfile('*.mat', 'Pick an ACF detector file');
                if isequal(filename, 0),
                    return;
                end
                filename = fullfile(pathname, filename);
            end
            
            self.acf_detector = AcfDetector(filename);
        end
            
        function load_classifier (self, filename)
            % LOAD_CLASSIFIER (self, filename)
            %
            % Loads classifier from file
            
            if ~exist('filename', 'var') || isempty(filename),
                [ filename, pathname ] = uigetfile('*.mat', 'Pick a classifier file');
                if isequal(filename, 0),
                    return;
                end
                filename = fullfile(pathname, filename);
            end
            
            tmp = load(filename);
            self.svm_classifier = tmp.classifier;
        end
        
        function detections = process_image (self, image_filename, varargin)
            % detections = PROCESS_IMAGE (self, image_filename, varargin)
            %
            % Processes the given input image.
            %
            % Input:
            %  - self: @PolypDetector instance
            %  - image_filename: name of the input image
            %  - varargin: additional key/value pairs:
            %    - cache_dir: path to cache directory
            %    - regions_only: boolean indicating whether to detect only
            %      region proposals (first stage) or final detections (full
            %      pipeline). Default: false
            %    - display_regions: whether to visualize detected regions
            %      or not (default: false)
            %    - display_detections: whether to visualize obtained
            %      detections (default: false)
            %    - display_detections_as_points: whether to visualize 
            %      obtained detections as points (default: false)
            %    - overlap_threshold: overlap threshold used when declaring
            %      a region as positive or negative in visualization
            %      (default: use evaluation_overlap setting)
            parser = inputParser();
            parser.addParameter('cache_dir', '', @ischar);
            parser.addParameter('regions_only', false, @islogical);
            parser.addParameter('display_regions', false, @islogical);
            parser.addParameter('display_detections', false, @islogical);
            parser.addParameter('display_detections_as_points', false, @islogical);
            parser.addParameter('overlap_threshold', self.evaluation_overlap, @isnumeric);
            parser.parse(varargin{:});
            
            cache_dir = parser.Results.cache_dir;
            regions_only = parser.Results.regions_only;
            display_regions = parser.Results.display_regions;
            display_detections = parser.Results.display_detections;
            display_detections_as_points = parser.Results.display_detections_as_points;
            overlap_threshold = parser.Results.overlap_threshold;
            
            %% Load and prepare the image
            [ I, basename, poly, annotations, annotations_pts ] = self.load_data(image_filename);
            
            % Mask the image
            mask = poly2mask(poly(:,1), poly(:,2), size(I,1), size(I,2));
            mask = imgaussfilt(double(mask), 2);
            Im = uint8( bsxfun(@times, double(I), mask) );
            
            % Crop the image
            xmin = min(poly(:,1));
            xmax = max(poly(:,1));
            ymin = min(poly(:,2));
            ymax = max(poly(:,2));
            Im = Im(ymin:ymax, xmin:xmax, :);
                        
            %% *** 1st stage: region proposal ***
            %% Run ACF detector
            if ~isempty(cache_dir),
                acf_cache_file = fullfile(cache_dir, 'acf-cache', [ basename, '.mat' ]);
            else
                acf_cache_file = '';
            end
            
            % Detect
            regions = self.detect_candidate_regions(Im, acf_cache_file);
            
            % Undo the effect of the crop
            regions(:,1) = regions(:,1) + xmin;
            regions(:,2) = regions(:,2) + ymin;
            
            % Display ACF regions
            if display_regions,
                visualize_detections_or_regions(I, poly, annotations, regions, 'multiple_matches', true, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: ACF', basename));
            end
            
            if regions_only,
                detections = regions;
                return;
            end
            
            %% *** 2nd stage: region classification ***
            % Make sure we have an SVM classifier ready
            assert(~isempty(self.svm_classifier), 'Invalid SVM classifier; cannot classify regions!');
            
            %% Extract CNN features from detected regions
            % Make sure to extract from original image, and not masked one!
            if ~isempty(cache_dir),
                cnn_cache_file = fullfile(cache_dir, 'cnn-cache', [ basename, '.mat' ]);
            else
                cnn_cache_file = '';
            end
            
            % Extract
            features = self.extract_features_from_regions(I, regions, cnn_cache_file);
            
            %% Classify with SVM
            fprintf('Performing SVM classification...\n');
            [ ~, scores, ~ ] = self.svm_classifier.predict(features);
            
            positive_mask = scores > 0;
            positive_regions = regions(positive_mask,1:4);
            positive_scores = scores(positive_mask);
            
            %% Additional NMS on top of SVM predictions
            fprintf('Performing NMS on top of SVM predictions...\n');
            
            detections = bbNms([ positive_regions, positive_scores' ], ...
                'type', 'maxg', ...
                'overlap', self.svm_nms_overlap, ...
                'ovrDnm', 'union');
                        
            % Display detections
            if display_detections,
                visualize_detections_or_regions(I, poly, annotations, detections, 'multiple_matches', false, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: Final', basename));
            end
            
            % Display detection points
            if display_detections_as_points,
                visualize_detections_as_points(I, poly, annotations_pts, detections, 'prefix', sprintf('%s: Final points', basename));
            end
        end
        
        function svm = train_svm_classifier (self, varargin)
            parser = inputParser();
            parser.addParameter('cache_dir', '', @ischar);
            parser.addParameter('train_images', self.default_train_images, @iscell);
            parser.addParameter('display_svm_samples', false, @islogical);
            parser.parse(varargin{:});
            
            cache_dir = parser.Results.cache_dir;
            train_images = parser.Results.train_images;
            display_svm_samples = parser.Results.display_svm_samples;
            
            %% Process all train images to get the features and boxes
            num_images = numel(train_images);
            
            all_features = cell(1, num_images);
            all_labels = cell(1, num_images);
            for i = 1:num_images,
                image_file = train_images{i};
                fprintf('Processing train image #%d/%d: %s\n', i, num_images, train_images{i});
                
                % Detect regions in the image
                [ I, basename, poly, annotations ] = self.load_data(image_file);
                regions = self.process_image(image_file, 'regions_only', true, 'cache_dir', cache_dir);

                % Extract CNN features
                % NOTE: we extract features from all regions, even the ones
                % that will be later discarded, in order to keep cache
                % files consistent with those produced by the
                % process_image() function!
                if ~isempty(cache_dir),
                    cnn_cache_file = fullfile(cache_dir, 'cnn-cache', [ basename, '.mat' ]);
                else
                    cnn_cache_file = '';
                end
                features = self.extract_features_from_regions(I, regions, cnn_cache_file);
                
                % Determine whether boxes are positive or negative
                mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
                [ ~, regions_pos ] = evaluate_detections(regions, annotations, 'threshold', self.training_positive_overlap, 'multiple', true, 'validity_mask', mask);
                [ ~, regions_neg ] = evaluate_detections(regions, annotations, 'threshold', self.training_negative_overlap, 'multiple', true, 'validity_mask', mask);
                
                % Boxes must be strictly positive or negative (this will
                % filter out the ones that were marked as "ignore", as well
                % as those whose overlap was between negative and positive
                % overlap)
                valid_mask = regions_pos(:,6) == 1 | regions_neg(:,6) == 0; 
                
                regions(~valid_mask, :) = [];
                features(:,~valid_mask) = [];
                labels = regions_pos(valid_mask, 6); % Doesn't matter if we take positive or negative output...
                
                % Add to the output
                all_features{i} = features;
                all_labels{i} = 2*labels - 1;
                
                %% Visualize negative samples
                if display_svm_samples,
                    fig = figure('Name', sprintf('SVM training samples: %s', basename));
                    clf(fig);

                    % Show image
                    Im = uint8( bsxfun(@times, double(I), 0.50*mask + 0.50) );
                    imshow(Im);
                    hold on;

                    % Draw chosen regions
                    draw_boxes(regions(labels == 1,:), fig, 'color', 'green', 'line_style', '-'); % Positive
                    draw_boxes(regions(labels == 0,:), fig, 'color', 'red', 'line_style', '-'); % Negative
                    
                    % Create fake plots for legend entries
                    h = zeros(1,2);
                    h(1) = plot([0,0], [0,0], '-', 'Color', 'green', 'LineWidth', 2);
                    h(2) = plot([0,0], [0,0], '-', 'Color', 'red', 'LineWidth', 2);
                    legend(h, 'pos', 'neg');
                    
                    drawnow();
                end
            end
            
            %% Train the SVM
            % Gather 
            all_features = horzcat( all_features{:} );
            all_labels = vertcat( all_labels{:} );
            
            fprintf('Training SVM with %d samples, %d positive (%.2f%%), %d negative (%.2f%%)\n', numel(all_labels), sum(all_labels==1), 100*sum(all_labels==1)/numel(all_labels), sum(all_labels==-1), 100*sum(all_labels==-1)/numel(all_labels));
            
            % Train
            svm = self.svm_create_function(); % Create SVM
            svm.train(all_features, all_labels);
            
            if nargout < 1,
                self.svm_classifier = svm;
            end
        end
    end
    
    %% Presets
    % These are the pre-defined detector pipelines, provided for
    % convenience
    methods (Static)
        function self = preset_mnist_fc1 ()
            % self = PRESET_MNIST_FC1 ()
            %
            % Creates a polyp detector pipeline with default ACF detector
            % and FC1 MNIST LeNet feature extractor.
            
            root_dir = fileparts(mfilename('fullpath'));
            
            % ACF detector
            acf_detector_name = fullfile(root_dir, 'detector', 'acf-polyp-default.mat');
            
            % Arguments for CNN feature extractor
            cnn_dir = fullfile(root_dir, 'detector', 'cnn-mnist');
            
            cnn_arguments = { ...
                fullfile(cnn_dir, 'lenet.prototxt'), ...
                fullfile(cnn_dir, 'lenet_iter_10000.caffemodel'), ...
                'layer_name', 'ip1', ...
                'pixel_scale', 1/256, ...
                'use_gpu', true ...
            };

            % Create
            self = PolypDetector('acf_detector', acf_detector_name, 'cnn_arguments', cnn_arguments);
        end
        
        function self = preset_mnist_proba ()
            % self = PRESET_MNIST_PROBA ()
            %
            % Creates a polyp detector pipeline with default ACF detector
            % and PROBA MNIST LeNet feature extractor.
            
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

            % Create
            self = PolypDetector('acf_detector', acf_detector_name, 'cnn_arguments', cnn_arguments);
        end
        
        function self = preset_imagenet_fc7 ()
            % self = PRESET_IMAGENET_FC7 ()
            %
            % Creates a polyp detector pipeline with default ACF detector
            % and FC7 ImageNet feature extractor.
            
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
        
            %% Create
            self = PolypDetector('acf_detector', acf_detector_name, 'cnn_arguments', cnn_arguments);
        end 
    end
    
    %% Default test and train images
    properties
        default_train_images = { ...
            'data/07.03.jpg', ...
            'data/13.01.jpg', ...
            'data/13.03.jpg', ...
            'data/13.04.jpg', ...
            'data/13.05.jpg' };
        
        default_test_images = {
            'data/01.01.jpg', ...
            'data/01.02.jpg', ...
            'data/01.03.jpg', ...
            'data/01.04.jpg', ...
            'data/01.05.jpg', ...
            'data/02.01.jpg', ...
            'data/02.02.jpg', ...
            'data/02.03.jpg', ...
            'data/02.04.jpg', ...
            'data/02.05.jpg', ...
            'data/03.01.jpg', ...
            'data/03.02.jpg', ...
            'data/03.03.jpg', ...
            'data/03.04.jpg', ...
            'data/03.05.jpg', ...
            'data/04.01.jpg', ...
            'data/04.02.jpg', ...
            'data/04.03.jpg', ...
            'data/04.04.jpg', ...
            'data/04.05.jpg', ...
            'data/05.01.jpg', ...
            'data/05.02.jpg', ...
            'data/05.03.jpg', ...
            'data/06.01.jpg', ...
            'data/06.02.jpg', ...
            'data/06.03.jpg', ...
            'data/07.01.jpg', ...
            'data/07.02.jpg', ...
            'data/08.01.jpg', ...
            'data/08.03.jpg' };
    end
end

%% Utility functions
function ensure_path_exists (filename)
    pathname = fileparts(filename);
    if ~exist(pathname, 'dir'),
        mkdir(pathname);
    end
end

function handles = draw_boxes (boxes, fig, varargin)
    parser = inputParser();
    parser.addParameter('color', 'red', @ischar);
    parser.addParameter('line_style', '-', @ischar);
    parser.addParameter('line_width', 1.0, @isnumeric);
    parser.parse(varargin{:});
    
    color = parser.Results.color;
    line_style = parser.Results.line_style;    
    line_width = parser.Results.line_width;    

    % Make figure current
    set(groot, 'CurrentFigure', fig);
            
    % [ x, y, w, h ] -> [ x1, y1, x2, y2 ]
    x1 = boxes(:,1)';
    y1 = boxes(:,2)';
    x2 = boxes(:,1)' + boxes(:,3)' + 1;
    y2 = boxes(:,2)' + boxes(:,4)' + 1;
            
    % Draw boxes
    handles = line([ x1, x1, x1, x2;
                     x2, x2, x1, x2 ], ...
                   [ y1, y2, y1, y1;
                     y1, y2, y2, y2 ], 'Color', color, 'LineStyle', line_style, 'LineWidth', line_width);
end

function boxes = enlarge_boxes (boxes, scale_factor)
    % boxes = ENLARGE_BOXES (boxes, scale_factor)
    
    % Load
    x = boxes(:,1);
    y = boxes(:,2);
    w = boxes(:,3);
    h = boxes(:,4);
    
    % Modify
    extra_width  = w * (scale_factor - 1);
    extra_height = h * (scale_factor - 1);
        
    x = x - extra_width/2;
    y = y - extra_height/2;
    w = w + extra_width;
    h = h + extra_height;
    
    % Store
    boxes(:,1) = x;
    boxes(:,2) = y;
    boxes(:,3) = w;
    boxes(:,4) = h;
end


function [ gt, dt ] = evaluate_detections (detections, annotations, varargin)
    parser = inputParser();
    parser.addParameter('threshold', 0.5, @isnumeric);
    parser.addParameter('multiple', false, @islogical);
    parser.addParameter('validity_mask', [], @islogical);
    parser.parse(varargin{:});
    
    threshold = parser.Results.threshold;
    multiple = parser.Results.multiple;
    validity_mask = parser.Results.validity_mask;

    % Agument annotations with "ignore" flag
    if isempty(validity_mask),
        invalid_idx = zeros(size(annotations, 1), 1);
    else
        x1 = annotations(:,1);
        y1 = annotations(:,2);
        x2 = annotations(:,1) + annotations(:,3) + 1;
        y2 = annotations(:,2) + annotations(:,4) + 1;
        
        x1 = max(min(round(x1), size(validity_mask, 2)), 1);
        y1 = max(min(round(y1), size(validity_mask, 1)), 1);
        x2 = max(min(round(x2), size(validity_mask, 2)), 1);
        y2 = max(min(round(y2), size(validity_mask, 1)), 1);
        
        valid_tl = validity_mask( sub2ind(size(validity_mask), y1, x1) );
        valid_tr = validity_mask( sub2ind(size(validity_mask), y1, x2) );
        valid_bl = validity_mask( sub2ind(size(validity_mask), y2, x1) );
        valid_br = validity_mask( sub2ind(size(validity_mask), y2, x2) );
        
        invalid_idx = ~(valid_tl & valid_tr & valid_bl & valid_br);
    end
    annotations(:,5) = invalid_idx;
    
    % Evaluate using Dollar's toolbox
    [ gt, dt ] = bbGt('evalRes', annotations, detections, threshold, multiple);
end

function fig = visualize_detections_or_regions (I, polygon, annotations, detections, varargin)
    % fig = VISUALIZE_RESULTS (I, polygon, annotations, detections, varargin)

    parser = inputParser();
    parser.addParameter('fig', [], @ishandle);
    parser.addParameter('multiple_matches', false, @islogical);
    parser.addParameter('overlap_threshold', 0.3, @isnumeric);
    parser.addParameter('prefix', '', @ischar);
    parser.parse(varargin{:});

    fig = parser.Results.fig;
    multiple_matches = parser.Results.multiple_matches;
    overlap_threshold = parser.Results.overlap_threshold;
    prefix = parser.Results.prefix;

    % Create mask
    mask = poly2mask(polygon(:,1), polygon(:,2), size(I, 1), size(I,2));

    % Evaluate detections (if annotations are available)
    if ~isempty(annotations),
        [ gt, det ] = evaluate_detections(detections, annotations, 'threshold', overlap_threshold, 'multiple', multiple_matches, 'validity_mask', mask);
    end
    
    % Figure
    if isempty(fig),
        fig = figure();
    else
        set(groot, 'CurrentFigure', fig);
    end
    clf(fig);

    % Show image
    Im = uint8( bsxfun(@times, double(I), 0.50*mask + 0.50) );
    imshow(Im);
    hold on;

    if ~isempty(annotations),
        % Draw ground-truth; TN and FN
        draw_boxes(gt(gt(:,5) == 1,:), fig, 'color', 'cyan', 'line_style', '-'); % TP
        draw_boxes(gt(gt(:,5) == 0,:), fig, 'color', 'yellow', 'line_style', '-'); % FN
        draw_boxes(gt(gt(:,5) == -1,:), fig, 'color', 'magenta', 'line_style', '-'); % ignore

        % Draw detections; TP and FP
        draw_boxes(det(det(:,6) == 1,:), fig, 'color', 'green', 'line_style', '-'); % TP
        draw_boxes(det(det(:,6) == 0,:), fig, 'color', 'red', 'line_style', '-'); % FP

        % Create fake plots for legend entries
        h = [];
        h(end+1) = plot([0,0], [0,0], '-', 'Color', 'cyan', 'LineWidth', 2);
        h(end+1) = plot([0,0], [0,0], '-', 'Color', 'yellow', 'LineWidth', 2);
        h(end+1) = plot([0,0], [0,0], '-', 'Color', 'green', 'LineWidth', 2);
        h(end+1) = plot([0,0], [0,0], '-', 'Color', 'red', 'LineWidth', 2);
        h(end+1) = plot([0,0], [0,0], '-', 'Color', 'magenta', 'LineWidth', 2);
        legend(h, 'TP (annotated)', 'FN', 'TP (det)', 'FP', 'ignore');

        % Count
        tp = sum( gt(:,5) == 1 );
        fn = sum( gt(:,5) == 0 );
        %tp = sum( det(:,6) == 1 );
        fp = sum( det(:,6) == 0 );

        precision = 100*tp/(tp+fp);
        recall = 100*tp/(tp+fn);

        num_annotated = sum(gt(:,5) ~= -1);
        num_detected = sum(det(:,6) ~= -1);

        % Set title
        if ~isempty(prefix),
            prefix = sprintf('%s: ', prefix);
        end
        title = sprintf('%srecall: %.2f%%, precision: %.2f%%; counted: %d, annotated: %d ', prefix, recall, precision, num_detected, num_annotated);
    else
        draw_boxes(detections, fig, 'color', 'green', 'line_style', '-'); % TP
        title = sprintf('%s: num detected: %d', prefix, size(detections, 1));
    end
    
    set(fig, 'Name', title);
    
    % Display as text as well
    h = text(0, 0, title, 'Color', 'white', 'FontSize', 20, 'Interpreter', 'none');
    h.Position(1) = size(I, 2)/2 - h.Extent(3)/2;
    h.Position(2) = h.Extent(4);
        
    % Draw
    drawnow();
end

function fig = visualize_detections_as_points (I, polygon, annotations, detections, varargin)
    % fig = VISUALIZE_RESULTS (I, polygon, annotations, detections, varargin)

    parser = inputParser();
    parser.addParameter('fig', [], @ishandle);
    parser.addParameter('prefix', '', @ischar);
    parser.parse(varargin{:});

    fig = parser.Results.fig;
    prefix = parser.Results.prefix;

    % Create mask
    mask = poly2mask(polygon(:,1), polygon(:,2), size(I, 1), size(I,2));

    if isempty(fig),
        fig = figure();
    else
        set(groot, 'CurrentFigure', fig);
    end
    clf(fig);

    % Show image
    Im = uint8( bsxfun(@times, double(I), 0.50*mask + 0.50) );

    imshow(Im);
    hold on;
    
    % Draw manual annotations
    h = [];
    legend_entries = {};

    if ~isempty(annotations),
        num_annotations = size(annotations, 1);
        colors = lines(num_annotations);
        for i = 1:num_annotations,
            annotation_id = annotations{i, 1};
            annotation_points = annotations{i, 2};
            
            h(end+1) = plot(annotation_points(:,1), annotation_points(:,2), 'ko', 'MarkerFaceColor', colors(i,:));
            legend_entries{end+1} = sprintf('%s (%d)', annotation_id, size(annotation_points, 1));
        end
    end

    % Draw detections
    detection_points = detections(:,1:2) + detections(:,3:4)/2;
    h(end+1) = plot(detection_points(:,1), detection_points(:,2), 'gx', 'MarkerSize', 8, 'LineWidth', 2);
    legend_entries{end+1} = sprintf('Detector (%d)', size(detections, 1));
    
    % Legend
    legend(h, legend_entries, 'Location', 'NorthEast', 'Interpreter', 'none');
    
    % Set title    
    title = prefix;
    set(fig, 'Name', title);
    
    % Display as text as well
    h = text(0, 0, title, 'Color', 'white', 'FontSize', 20, 'Interpreter', 'none');
    h.Position(1) = size(I, 2)/2 - h.Extent(3)/2;
    h.Position(2) = h.Extent(4);
    
    % Draw
    drawnow();
end
