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
        training_positive_overlap = 0.5
        training_negative_overlap = 0.1
        
        % L2-normalize feature vectors
        l2_normalized_features = true
        
        % Non-maxima suppression overlap threshold for confirmed detections
        svm_nms_overlap = 0.1
        
        % Evaluation overlap
        evaluation_overlap = 0.1
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
        
    % Processing pipeline steps (generally not meant to be used outside
    % this class)
    methods
        function [ regions, time_det, time_nms ] = detect_candidate_regions (self, I, cache_file)
            % [ regions, time_det, time_nms ] = DETECT_CANDIDATE_REGIONS (self, I, cache_file)
            %
            % Detects candidate regions in the given image.
            %
            % Input:
            %  - self:
            %  - I: image
            %  - cache_file: optional cache file to use (default: '')
            %
            % Output:
            %  - regions: detected regions
            %  - time_det: time spent in region detection
            %  - time_nms: time spent in the first non-maxima suppression
            %    pass
            %
            % Note: creates ACF instance on demand
            
            if ~exist('cache_file', 'var'),
                cache_file = '';
            end
            
            %% Region detection / caching
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
                % Create ACF detector, if necessary
                if isempty(self.acf_detector),
                    self.acf_detector = self.acf_detector_factory();
                end
                
                % Run ACF detector
                [ regions, regions_all, time_det, time_nms ] = self.acf_detector.detect(I, 'nms_overlap', self.acf_nms_overlap);
                
                % Save to cache
                if ~isempty(cache_file),
                    vicos.utils.ensure_path_exists(cache_file);
                    
                    tmp = struct(...
                        'nms_overlap', self.acf_nms_overlap, ...
                        'regions', regions, ...
                        'regions_all', regions_all, ...
                        'time_det', time_det, ...
                        'time_nms', time_nms); %#ok<NASGU>
                    
                    save(cache_file, '-v7.3', '-struct', 'tmp');
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
            %  - self:
            %  - I: image
            %  - regions: Nx4 matrix describing regions
            %  - cache_file: optional cache file to use (default: '')
            %
            % Output:
            %  - features: DxN matrix of extracted features
            %  - time: time spent in feature extraction
            %
            % Note: creates CNN Feature Extractor instance on demand
            
            if ~exist('cache_file', 'var'),
                cache_file = '';
            end
            
            %% Feature extraction / caching
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
            
                % Create extractor, if necessary
                if isempty(self.cnn_extractor),
                    self.cnn_extractor = self.feature_extractor_factory();
                end
                
                % Extract CNN features
                [ features, time ] = self.cnn_extractor.extract(I, 'regions', boxes);
                
                % Save to cache
                if ~isempty(cache_file),
                    vicos.utils.ensure_path_exists(cache_file);
                    
                    tmp = struct(...
                        'features', features, ...
                        'time', time); %#ok<NASGU>
                    
                    save(cache_file, '-v7.3', '-struct', 'tmp');
                end
            end
            
            %% Feature normalization
            if self.l2_normalized_features,
                features = bsxfun(@rdivide, features, sqrt(sum(features .^ 2)));
            end
        end
    end

    methods
        function train_and_evaluate (self, result_dir, varargin)
            % TRAIN_AND_EVALUATE (self, result_dir, varargin)
            %
            % Input:
            %  - self:
            %  - result_dir:
            %  - varargin: optional key/value pairs
            %     - train_images: cell array of train image filenames
            %     - test_images: cell array of test image filenames
            %     - display_svm_samples: boolean indicating whether to
            %       visualize SVM training samples or not
            
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
            
            %% Train SVM
            classifier_file = fullfile(result_dir, 'classifier.mat');
            if exist(classifier_file, 'file'),
                % Load from file
                tmp = load(classifier_file);
                self.svm_classifier = tmp.classifier;
            else
                % Train
                t = tic();
                self.svm_classifier = self.train_svm_classifier('train_images', train_images, 'cache_dir', cache_dir, 'display_svm_samples', display_svm_samples);
                time = toc(t);
                    
                % Save
                vicos.utils.ensure_path_exists(classifier_file);
                
                tmp = struct(...
                    'classifier', self.svm_classifier, ...
                    'time', time); %#ok<NASGU>
                
                save(classifier_file, '-v7.3', '-struct', 'tmp');
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
                                  'num_detected', 0), 1, numel(test_images));
                              
            for i = 1:numel(test_images),
                test_image = test_images{i};
                
                fprintf('Processing test image #%d/%d: %s\n', i, numel(test_images), test_image);
                
                %% Load test image
                [ I, basename, poly, annotations ] = self.load_data(test_image);

                % Try loading result from cache
                results_file = fullfile(result_dir, [ basename, '.mat' ]);
                if exist(results_file, 'file'),
                    all_results(i) = load(results_file);
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
                
                vicos.utils.ensure_path_exists(results_file);
                save(results_file, '-v7.3', '-struct', 'results');
                
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
        end
        
        function leave_one_out_cross_validation (self, result_dir)
            % LEAVE_ONE_OUT_CROSS_VALIDATION (self, result_dir)
                        
            % Cache 
            cache_dir = fullfile(result_dir, 'cache');
            
            % Create list of images
            images = union(self.default_train_images, self.default_test_images);
                        
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
                    all_results(i) = load(results_file);
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
                    self.svm_classifier = self.train_svm_classifier('train_images', train_images, 'cache_dir', cache_dir);
                    time = toc(t);
                    
                    % Save
                    vicos.utils.ensure_path_exists(classifier_file);
                    
                    tmp = struct(...
                        'classifier', self.svm_classifier, ...
                        'time', time); %#ok<NASGU>
                    
                    save(classifier_file, '-v7.3', '-struct', 'tmp');
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
                
                vicos.utils.ensure_path_exists(results_file);
                save(results_file, '-v7.3', '-struct', 'results');
                
                all_results(i) = results;
            end
            
            save(fullfile(result_dir, 'all_results.mat'), '-v7.3', 'all_results');
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
            
            self.acf_detector = vicos.AcfDetector(filename);
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
            %
            % Output:
            %  - detections
            
            % Input arguments
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
            [ ~, scores ] = self.svm_classifier.predict(features);
            
            positive_mask = scores >= 0;
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
            % svm = TRAIN_SVM_CLASSIFIER (self, varargin)
            %
            % Trains an SVM classifier.
            %
            % Input:
            %  - self:
            %  - varagin: optional key/value pairs:
            %     - cache_dir: cache directory (default: '')
            %     - train_images: cell array of train image file names
            %       (default: use built-in list)
            %     - display_svm_samples: visualize SVM training samples on
            %       each training image
            %
            % Output:
            %  - svm: trained SVM classifier
            
            % Input arguments
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
                
                %% Visualize training samples
                if display_svm_samples,
                    fig = figure('Name', sprintf('SVM training samples: %s', basename));
                    clf(fig);

                    % Show image
                    Im = uint8( bsxfun(@times, double(I), 0.50*mask + 0.50) );
                    imshow(Im);
                    hold on;

                    % Draw chosen regions
                    vicos.utils.draw_boxes(regions(labels == 1,:), 'color', 'green', 'line_style', '-'); % Positive
                    vicos.utils.draw_boxes(regions(labels == 0,:), 'color', 'red', 'line_style', '-'); % Negative
                    
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
            svm = self.classifier_factory(); % Create SVM
            svm.train(all_features, all_labels);
            
            if nargout < 1,
                self.svm_classifier = svm;
            end
        end
    end
    
    methods (Static)
        function root_dir = get_root_code_path ()
            root_dir = fileparts(mfilename('fullpath'));
        end
        
        function [ I, basename, poly, boxes, manual_annotations ] = load_data (image_filename)
            % [ I, basename, poly, boxes, manual_annotations ] = LOAD_DATA (self, image_filename)
            %
            % Loads an image and, if available, its accompanying polygon, 
            % bounding box, and point-wise annotations.
            %
            % Input:
            %  - image_filename: input image filename
            %
            % Output:
            %  - I: loaded image
            %  - basename: image's basename, which can be passed to
            %    subsequent processing functions for data caching
            %  - poly: polygon that describes ROI
            %  - boxes: manually-annotated bounding boxes
            %  - manual_annotations: a cell array of point-wise manual
            %    annotations
            
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
    end
    
    
    %% Feature extractor presets
    methods (Static)
        function extractor = feature_extractor_imagenet_fc7 ()
            % extractor = FEATURE_EXTRACTOR_IMAGENET_FC7 ()
            %
            % Creates a ImageNet FC7 CNN feature extractor.
            
            % Arguments for CNN feature extractor
            cnn_dir = fullfile(PolypDetector.get_root_code_path(), 'detector', 'cnn-rcnn');
            
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

%% Utility functions
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
    % fig = VISUALIZE_DETECTIONS_OR_REGIONS (I, polygon, annotations, detections, varargin)
    %
    % Visualizes detection/region-proposal bounding boxes.

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
        vicos.utils.draw_boxes(gt(gt(:,5) == 1,:), 'color', 'cyan', 'line_style', '-'); % TP
        vicos.utils.draw_boxes(gt(gt(:,5) == 0,:), 'color', 'yellow', 'line_style', '-'); % FN
        vicos.utils.draw_boxes(gt(gt(:,5) == -1,:), 'color', 'magenta', 'line_style', '-'); % ignore

        % Draw detections; TP and FP
        vicos.utils.draw_boxes(det(det(:,6) == 1,:), 'color', 'green', 'line_style', '-'); % TP
        vicos.utils.draw_boxes(det(det(:,6) == 0,:), 'color', 'red', 'line_style', '-'); % FP

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
        vicos.utils.draw_boxes(detections, 'color', 'green', 'line_style', '-'); % TP
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
    % fig = VISUALIZE_DETECTIONS_AS_POINTS (I, polygon, annotations, detections, varargin)
    %
    % Visualizes detections' centroids.

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
