classdef PolypDetector < handle
    % POLYPDETECTOR - Polyp detector class
    %
    % (C) 2016 Rok Mandeljc <rok.mandeljc@fri.uni-lj.si>
    
    properties
        cache_dir
        
        % Detection pipeline components
        acf_detector
        cnn_extractor
        svm_classifier
    end
    
    properties (Access = private)
        cnn_arguments
    end
    
    methods
        function self = PolypDetector (varargin)
            parser = inputParser();
            parser.addParameter('acf_detector', 'detector/acf-polyp-default.mat', @ischar);
            parser.addParameter('cnn_arguments', {}, @iscell);
            parser.addParameter('cache_dir', '', @ischar);
            parser.parse(varargin{:});
            
            % Cache path
            self.cache_dir = parser.Results.cache_dir;
            
            % Create ACF detector wrapper
            acf_detector_file = parser.Results.acf_detector;
            self.acf_detector = AcfDetector(acf_detector_file);
            
            % Create CNN feature extractor
            self.cnn_arguments = parser.Results.cnn_arguments;
            self.cnn_extractor = CnnFeatureExtractor(self.cnn_arguments{:});
        end
        
        
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
        
        function regions = detect_candidate_regions (self, I, varargin)
            % regions = DETECT_CANDIDATE_REGIONS (self, I, varargin)
            %
            % Detects candidate regions in the given image.
            %
            % Input:
            %  - self: @PolypDetector instance
            %  - I: image
            %  - basename: image basename, used for caching
            %
            % Output:
            %  - regions:
            
            parser = inputParser();
            parser.addParameter('basename', '', @ischar);
            parser.addParameter('nms_overlap', 0.5, @isnumeric);
            parser.parse(varargin{:});
            
            basename = parser.Results.basename;
            nms_overlap = parser.Results.nms_overlap;
            
            % Run ACF detector
            acf_cache_file = '';
            if ~isempty(self.cache_dir),
                acf_cache_file = fullfile(self.cache_dir, sprintf('acf_%s_nms-%g', self.acf_detector.name, nms_overlap), [ basename, '.mat' ]);
            end
            
            if ~isempty(acf_cache_file) && exist(acf_cache_file, 'file'),
                tmp = load(acf_cache_file);
                regions = tmp.regions;
            else
                [ regions, regions_all, time_det, time_nms ] = self.acf_detector.detect(I, 'nms_overlap', nms_overlap);
                
                if ~isempty(acf_cache_file),
                    % Save to cache file
                    ensure_path_exists(acf_cache_file);
                    save(acf_cache_file, 'regions', 'regions_all', 'time_det', 'time_nms');
                end
            end
        end
        
        function features = extract_features_from_regions (self, I, regions, basename)
            % features = EXTRACT_FEATURES_FROM_REGIONS (self, I, regions, basename)
            %
            % Extract CNN features from given regions.
            %
            % Input:
            %  - self: @PolypDetector instance
            %  - I: image
            %  - regions: Nx4 matrix describing regions
            %  - basename: image basename, used for caching
            %
            % Output:
            %  - features: DxN matrix of extracted features
            
            if ~exist('basename', 'var'),
                basename = '';
            end
            
            features_cache_file = '';
            if ~isempty(self.cache_dir),
                features_cache_file = fullfile(self.cache_dir, 'cnn', [ basename, '.mat' ]);
            end           

            if ~isempty(features_cache_file) && exist(features_cache_file, 'file'),
                tmp = load(features_cache_file);
                features = tmp.features;
            else
                % Convert [ x, y, w, h ] to [ x1, y1, x2, y2 ], in 4xN format
                boxes = [ regions(:,1), regions(:,2), regions(:,1)+regions(:,3)+1, regions(:,2)+regions(:,4)+1 ]';
            
                % Extract CNN features
                [ features, time ] = self.cnn_extractor.extract(I, 'regions', boxes);
                
                if ~isempty(features_cache_file),
                    % Save to cache file
                    ensure_path_exists(features_cache_file);
                    save(features_cache_file, 'features', 'time');
                end
            end
        end
        
        
        
        
        function detections = process_image (self, image_filename)
            %% Load and prepare the image
            [ I, basename, poly, annotations ] = self.load_data(image_filename);
            
            % Mask the image
            Im = mask_image_with_polygon(I, poly);
            
            %% Run ACF detector
            regions = self.detect_candidate_regions(Im, 'basename', basename);
            
            % Evaluate ACF detector
            if true,
                mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
                
                [ gt, det ] = evaluate_detections(regions, annotations, 'threshold', 0.5, 'multiple', false, 'validity_mask', mask);
                fig = figure('Name', 'ACF detection results');
                imshow(Im); hold on;
                
                % Draw ground-truth; TN and FN
                draw_boxes(gt(gt(:,5) == 1,:), fig, 'color', 'cyan', 'line_style', '-'); % TN
                draw_boxes(gt(gt(:,5) == 0,:), fig, 'color', 'yellow', 'line_style', '-'); % FN
                draw_boxes(gt(gt(:,5) == -1,:), fig, 'color', 'magenta', 'line_style', '-'); % ignore
                
                % Draw detections; TP and FP
                draw_boxes(det(det(:,6) == 1,:), fig, 'color', 'green', 'line_style', '-'); % TP
                draw_boxes(det(det(:,6) == 0,:), fig, 'color', 'red', 'line_style', '-'); % FP
                
                % Count
                tn = sum( gt(:,5) == 1 );
                fn = sum( gt(:,5) == 0 );
                tp = sum( det(:,6) == 1 );
                fp = sum( det(:,6) == 0 );
                
                precision = 100*tp/(tp+fp);
                recall = 100*tp/(tp+fn);
                
                set(fig, 'Name', sprintf('ACF: recall: %.2f%%, precision: %.2f%% ', recall, precision));
            end
            
            %% Extract CNN features from detected regions
            % Make sure to extract from original image, and not masked one!
            features = self.extract_features_from_regions(I, regions);
            
            %% Classify with SVM
            fprintf('Performing SVM classification...\n');
            t = tic();
            [ labels, scores, probabilities ] = self.svm_classifier.predict(features);
            fprintf(' > Done in %f seconds!\n', toc(t));
            
            %% Additional NMS on top of SVM predictions
            fprintf('Performing NMS on top of SVM predictions...\n');
            
            detections = bbNms([ regions, scores' ], ...
                'type', 'maxg', ...
                'overlap', 0.50, ...
                'ovrDnm', 'min');
            
            fprintf(' > Done in %f seconds!\n', toc(t));            
        end
        
        function train_svm_classifier (self)
            train_images = self.default_train_images;
            num_images = numel(train_images);
            
            for i = 1:num_images,
                image_file = train_images{i};
                fprintf('Processing train image #%d/%d: %s\n', i, num_images, train_images{i});
                
                % Load image
                [ I, basename, poly, annotations ] = self.load_data(image_file);
                
                % Mask image
                Im = mask_image_with_polygon(I, poly);
                
                % Run ACF
                regions = self.detect_candidate_regions(Im, 'basename', basename);
                
                % Determine whether boxes are positive or negative
                
                % Extract CNN features
            end
        end
        
        function [ tn, fn, tp, fp ] = evaluate_on_single_image (self, image_file, varargin)
            parser = inputParser();
            parser.addParameter('nms_overlap', 0.5, @isnumeric);
            parser.addParameter('eval_overlap', 0.5, @isnumeric);
            parser.addParameter('visualize', false, @islogical);
            parser.parse(varargin{:});
            
            nms_overlap = parser.Results.nms_overlap;
            eval_overlap = parser.Results.eval_overlap;
            visualize = parser.Results.visualize;
            
            %% Process image
            % Load image
            [ I, basename, poly, annotations ] = self.load_data(image_file);
                
            % Mask image
            Im = mask_image_with_polygon(I, poly);
                
            % Run ACF
            regions = self.detect_candidate_regions(Im, 'basename', basename, 'nms_overlap', nms_overlap);
                
            %% Evaluate
            mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
                
            [ gt, det ] = evaluate_detections(regions, annotations, 'threshold', eval_overlap, 'multiple', true, 'validity_mask', mask);
            
            tn = sum( gt(:,5) == 1 );
            fn = sum( gt(:,5) == 0 );
            tp = sum( det(:,6) == 1 );
            fp = sum( det(:,6) == 0 );
            
            accuracy = (tp + tn) / (tp + fp + fn + tn);
            precision = tp / (tp + fp);
            recall = tp / (tp + fn);
            
            %% Visualize
            if nargout == 0 || visualize,
                fig = figure('Name', sprintf('%s: accuracy: %.2f%%, recall: %.2f%%, precision: %.2f%%', basename, 100*accuracy, 100*recall, 100*precision));
                
                imshow(Im); hold on;
                
                % Draw ground-truth; TN and FN
                draw_boxes(gt(gt(:,5) == 1,:), fig, 'color', 'cyan', 'line_style', '-'); % TN
                draw_boxes(gt(gt(:,5) == 0,:), fig, 'color', 'yellow', 'line_style', '-'); % FN
                draw_boxes(gt(gt(:,5) == -1,:), fig, 'color', 'magenta', 'line_style', '-'); % ignore
                
                % Draw detections; TP and FP
                draw_boxes(det(det(:,6) == 1,:), fig, 'color', 'green', 'line_style', '-'); % TP
                draw_boxes(det(det(:,6) == 0,:), fig, 'color', 'red', 'line_style', '-'); % FP
            end
        end
        
        function evaluate_region_proposals (self, varargin)
            parser = inputParser();
            parser.addParameter('test_images', self.default_test_images, @iscell);
            parser.addParameter('nms_overlap', 0.5, @isnumeric);
            parser.addParameter('eval_overlap', 0.5, @isnumeric);
            parser.parse(varargin{:});
            
            test_images = parser.Results.test_images;
            nms_overlap = parser.Results.nms_overlap;
            eval_overlap = parser.Results.eval_overlap;
            
            fprintf('*** Evaluating ACF region proposals ***\n');
            fprintf('NMS overlap: %g\n', nms_overlap);
            fprintf('evaluation overlap: %g\n', eval_overlap);
            fprintf('\n');
            
            %% Process
            num_images = numel(test_images);
            
            % Allocate variables
            tp = nan(1, num_images);
            fp = nan(1, num_images);
            tn = nan(1, num_images);
            fn = nan(1, num_images);
            
            for i = 1:num_images,
                image_file = test_images{i};
                fprintf('Processing test image #%d/%d: %s\n', i, num_images, test_images{i});
                
                [ tn(i), fn(i), tp(i), fp(i) ] = self.evaluate_on_single_image(image_file, 'nms_overlap', nms_overlap, 'eval_overlap', eval_overlap);
            end
            
            tn = sum(tn);
            tp = sum(tp);
            fn = sum(fn);
            fp = sum(fp);
            
            accuracy = (tp + tn) / (tp + fp + fn + tn);
            precision = tp / (tp + fp);
            recall = tp / (tp + fn);
            
            fprintf('\n');
            fprintf('Accuracy: %.2f%%\n', 100*accuracy);
            fprintf('Recall: %.2f%%\n', 100*recall);
            fprintf('Precision: %.2f%%\n', 100*precision);
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
