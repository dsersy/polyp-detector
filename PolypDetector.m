classdef PolypDetector < handle
    % POLYPDETECTOR - Polyp detector class
    %
    % (C) 2016 Rok Mandeljc <rok.mandeljc@fri.uni-lj.si>
    
    properties
        % Detection pipeline components
        acf_detector
        cnn_extractor
        svm_classifier
    end
    
    properties (Access = private)
        cnn_arguments
    end
    
    
    %% Public API
    methods
        function self = PolypDetector (varargin)
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
        
    % Processing pipeline steps
    methods
        function [ I, basename, poly, boxes ] = load_data (self, image_filename)
            % [ I, basename, poly, annotations ] = LOAD_DATA (self, image_filename)
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
            
            % Load annotations
            if nargout > 3,
                boxes_file = fullfile(pathname, [ basename, '.bbox' ]);
                if exist(boxes_file, 'file'),
                    boxes = load(boxes_file);
                else
                    boxes = [];
                end
            end
        end
        
        function [ regions, time_det, time_nms ] = detect_candidate_regions (self, I, varargin)
            % [ regions, time_det, time_nms ] = DETECT_CANDIDATE_REGIONS (self, I, varargin)
            %
            % Detects candidate regions in the given image.
            %
            % Input:
            %  - self: @PolypDetector instance
            %  - I: image
            %  - varargin: additional key/value pairs
            %    - cache_file: cache file to use, if provided
            %    - nms_overlap: the overlap threshold for the initial
            %      non-maxima suppression pass
            %
            % Output:
            %  - regions: detected regions
            %  - time_det: time spent in region detection
            %  - time_nms: time spent in the first non-maxima suppression
            %    pass
            
            parser = inputParser();
            parser.addParameter('cache_file', '', @ischar);
            parser.addParameter('nms_overlap', 0.5, @isnumeric);
            parser.parse(varargin{:});
            
            cache_file = parser.Results.cache_file;
            nms_overlap = parser.Results.nms_overlap;
            
            if ~isempty(cache_file) && exist(cache_file, 'file'),
                % Load from cache
                tmp = load(cache_file);
                
                % Validate cache file
                assert(nms_overlap == tmp.nms_overlap, 'Invalid cache file; nms_overlap mismatch!');
                
                % Copy from cache
                regions = tmp.regions;
                time_det = tmp.time_det;
                time_nms = tmp.time_nms;
            else
                % Run ACF detector
                [ regions, regions_all, time_det, time_nms ] = self.acf_detector.detect(I, 'nms_overlap', nms_overlap);
                
                % Save to cache
                if ~isempty(cache_file),
                    ensure_path_exists(cache_file);
                    save(cache_file, 'regions', 'regions_all', 'time_det', 'time_nms', 'nms_overlap');
                end
            end
        end
        
        function [ features, time ] = extract_features_from_regions (self, I, regions, varargin)
            % features = EXTRACT_FEATURES_FROM_REGIONS (self, I, regions)
            %
            % Extract CNN features from given regions.
            %
            % Input:
            %  - self: @PolypDetector instance
            %  - I: image
            %  - regions: Nx4 matrix describing regions
            %  - varargin: additional key/value pairs
            %    - cache_file: cache file to use, if provided
            %
            % Output:
            %  - features: DxN matrix of extracted features
            %  - time: time spent in feature extraction
            
            parser = inputParser();
            parser.addParameter('cache_file', '', @ischar);
            parser.parse(varargin{:});
            
            cache_file = parser.Results.cache_file;
            
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
                    save(cache_file, 'features', 'time');
                end
            end
        end
    end

    
    methods
        function fig = visualize_results (self, I, polygon, annotations, detections, varargin)
            % fig = VISUALIZE_RESULTS (self, I, polygon, annotations, detections, varargin)
            
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
            
            % Evaluate detections
            [ gt, det ] = evaluate_detections(detections, annotations, 'threshold', overlap_threshold, 'multiple', multiple_matches, 'validity_mask', mask);
            
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

            if ~isempty(prefix),
                prefix = sprintf('%s: ', prefix);
            end
            title = sprintf('%srecall: %.2f%%, precision: %.2f%%; counted: %d, annotated: %d ', prefix, recall, precision, num_detected, num_annotated);
            
            set(fig, 'Name', title);
            drawnow();
        end
    end

    
    methods
        function leave_one_out_cross_validation (self, result_dir)
            % LEAVE_ONE_OUT_CROSS_VALIDATION (self, result_dir)
            
            overlap_threshold = 0.3;
            
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
                    classifier_file = fullfile(result_dir, 'classifiers', [ basename, '.mat' ]);
                    ensure_path_exists(classifier_file);
                    save(classifier_file, 'classifier', 'time');
                end
                
                %% Process the left-out image
                detections = self.process_image(test_image, 'cache_dir', cache_dir);
                
                %% Evaluate
                % Create mask
                mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
            
                % Evaluate detections
                [ gt, det ] = evaluate_detections(detections, annotations, 'threshold', overlap_threshold, 'multiple', false, 'validity_mask', mask);
                
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
            %    - visualize_regions: whether to visualize detected regions
            %      or not (default: false)
            %    - visualize_detections: whether to visualize obtained
            %      detections (default: false)
            %    - overlap_threshold: overlap threshold used for
            %      visualization of region proposals/detections
            parser = inputParser();
            parser.addParameter('cache_dir', '', @ischar);
            parser.addParameter('regions_only', false, @islogical);
            parser.addParameter('visualize_regions', false, @islogical);
            parser.addParameter('visualize_detections', false, @islogical);
            parser.addParameter('overlap_threshold', 0.3, @isnumeric);
            parser.parse(varargin{:});
            
            cache_dir = parser.Results.cache_dir;
            regions_only = parser.Results.regions_only;
            visualize_regions = parser.Results.visualize_regions;
            visualize_detections = parser.Results.visualize_detections;
            overlap_threshold = parser.Results.overlap_threshold;
            
            %% Load and prepare the image
            [ I, basename, poly, annotations ] = self.load_data(image_filename);
            
            % Mask the image
            Im = mask_image_with_polygon(I, poly);
            
            %% *** 1st stage: region proposal ***
            %% Run ACF detector
            if ~isempty(cache_dir),
                acf_cache_file = fullfile(cache_dir, 'acf-cache', [ basename, '.mat' ]);
            else
                acf_cache_file = '';
            end
            
            % Detect
            regions = self.detect_candidate_regions(Im, 'cache_file', acf_cache_file);
            
            % Display ACF regions
            if visualize_regions,
                self.visualize_results(I, poly, annotations, regions, 'multiple_matches', true, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: ACF', basename));
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
            features = self.extract_features_from_regions(I, regions, 'cache_file', cnn_cache_file);
            
            %% Classify with SVM
            fprintf('Performing SVM classification...\n');
            [ labels, scores, probabilities ] = self.svm_classifier.predict(features);
            
            positive_mask = scores > 0;
            positive_regions = regions(positive_mask,1:4);
            positive_scores = scores(positive_mask);
            positive_proba = probabilities(positive_mask);
            
            %% Additional NMS on top of SVM predictions
            fprintf('Performing NMS on top of SVM predictions...\n');
            
            detections = bbNms([ positive_regions, positive_scores' ], ...
                'type', 'none', ...
                'overlap', 0.50, ...
                'ovrDnm', 'union');
                        
            % Display detections
            if visualize_detections,
                self.visualize_results(I, poly, annotations, detections, 'multiple_matches', false, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: Final', basename));
            end
        end
        
        function svm = train_svm_classifier (self, varargin)
            parser = inputParser();
            parser.addParameter('cache_dir', '', @ischar);
            parser.addParameter('positive_overlap', 0.5, @isscalar);
            parser.addParameter('train_images', self.default_train_images, @iscell);
            parser.addParameter('svm_function', @() classifier.LIBLINEAR());
            parser.parse(varargin{:});
            
            cache_dir = parser.Results.cache_dir;
            train_images = parser.Results.train_images;
            positive_overlap = parser.Results.positive_overlap;
            svm_function = parser.Results.svm_function;
            
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

                % Determine whether boxes are positive or negative
                mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
                [ ~, regions ] = evaluate_detections(regions, annotations, 'threshold', positive_overlap, 'multiple', true, 'validity_mask', mask);

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
                features = self.extract_features_from_regions(I, regions, 'cache_file', cnn_cache_file);

                % Determine labels, remove ignored regions
                labels = regions(:,6);
                
                invalid_mask = labels == -1;
                regions(invalid_mask, :) = [];
                
                labels = regions(:,6);
                features(:,invalid_mask) = [];
                
                % Add to the output
                all_features{i} = features;
                all_labels{i} = 2*labels - 1;
            end
            
            %% Train the SVM
            % Gather 
            all_features = horzcat( all_features{:} );
            all_labels = vertcat( all_labels{:} );
            
            fprintf('Training SVM with %d samples, %d positive (%.2f%%), %d negative (%.2f%%)\n', numel(all_labels), sum(all_labels==1), 100*sum(all_labels==1)/numel(all_labels), sum(all_labels==-1), 100*sum(all_labels==-1)/numel(all_labels));
            
            % Train
            svm = svm_function(); % Create SVM
            svm.train(all_features, all_labels);
            
            if nargout < 1,
                self.svm_classifier = svm;
            end
        end
    end
    
    % Default test and train images
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
