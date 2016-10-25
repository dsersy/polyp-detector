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
    %      or not (default: false); may be a figure handle (see note below)
    %    - display_regions_as_points: whether to visualize detected regions
    %      as points (default: false); may be a figure handle (see note 
    %      below)
    %    - display_detections: whether to visualize obtained
    %      detections (default: false); ; may be a figure handle (see note 
    %      below)
    %    - display_detections_as_points: whether to visualize
    %      obtained detections as points (default: false); may be a figure 
    %      handle (see note below)
    %    - overlap_threshold: overlap threshold used when declaring
    %      a detection as positive or negative in box visualization
    %      (default: use evaluation_overlap setting)
    %    - distance_threshold: distance threshold used when declaring a
    %      detection as positive or negative in point visualization
    %      (default: use the evaluation_distance setting)
    %
    % Output:
    %  - detections
    %
    % Note: instead of passing a logical value to display_ arguments, one
    % may pass a figure handle instead. In this case, the corresponding
    % visualization is drawn into the existing figure instead of a new one.
    
    % Input arguments
    parser = inputParser();
    parser.addParameter('cache_dir', '', @ischar);
    parser.addParameter('regions_only', false, @islogical);
    parser.addParameter('display_regions', false, @(x) islogical(x) || ishandle(x));
    parser.addParameter('display_regions_as_points', false, @(x) islogical(x) || ishandle(x));
    parser.addParameter('display_detections', false, @(x) islogical(x) || ishandle(x));
    parser.addParameter('display_detections_as_points', false, @(x) islogical(x) || ishandle(x));
    parser.addParameter('overlap_threshold', self.evaluation_overlap, @isnumeric);
    parser.addParameter('distance_threshold', self.evaluation_distance, @isnumeric);
    parser.addParameter('rescale_image', 1.0, @isnumeric);
    parser.parse(varargin{:});
    
    cache_dir = parser.Results.cache_dir;
    regions_only = parser.Results.regions_only;
    display_regions = parser.Results.display_regions;
    display_regions_as_points = parser.Results.display_regions_as_points;
    display_detections = parser.Results.display_detections;
    display_detections_as_points = parser.Results.display_detections_as_points;
    overlap_threshold = parser.Results.overlap_threshold;
    distance_threshold = parser.Results.distance_threshold;
    rescale_image = parser.Results.rescale_image;
    
    % Figures (because we allow figure handle to be passed via display_
    % parameters)
    if ishandle(display_regions),
        display_regions_fig = display_regions;
        display_regions = true;
    else
        display_regions_fig = [];
    end
    
    if ishandle(display_regions_as_points),
        display_regions_as_points_fig = display_regions_as_points;
        display_regions_as_points = true;
    else
        display_regions_as_points_fig = [];
    end
    
    if ishandle(display_detections),
        display_detections_fig = display_detections;
        display_detections = true;
    else
        display_detections_fig = [];
    end
    
    if ishandle(display_detections_as_points),
        display_detections_as_points_fig = display_detections_as_points;
        display_detections_as_points = true;
    else
        display_detections_as_points_fig = [];
    end    
    
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
    
    % Clamp
    xmin = max(round(xmin), 1);
    xmax = min(round(xmax), size(Im, 2));
    ymin = max(round(ymin), 1);
    ymax = min(round(ymax), size(Im, 1));
    
    Im = Im(ymin:ymax, xmin:xmax, :);
    
    % Upscale the image
    Im = imresize(Im, rescale_image);
    
    %% *** 1st stage: region proposal ***
    %% Run ACF detector
    if ~isempty(cache_dir),
        acf_cache_file = fullfile(cache_dir, 'acf-cache', sprintf('%s-scale_%g-acf_nms_%g.mat', basename, rescale_image, self.acf_nms_overlap));
    else
        acf_cache_file = '';
    end
    
    % Detect
    regions = self.detect_candidate_regions(Im, acf_cache_file);
    
    % Undo the scaling
    regions = regions / rescale_image;
    
    % Undo the effect of the crop
    regions(:,1) = regions(:,1) + xmin;
    regions(:,2) = regions(:,2) + ymin;
    
    % Display ACF regions
    if display_regions,
        self.visualize_detections_as_boxes(I, poly, annotations, regions, 'fig', display_regions_fig, 'multiple_matches', true, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: ACF', basename));
    end
    
    % Display ACF regions as points
    if display_regions_as_points,
        self.visualize_detections_as_points(I, poly, annotations_pts, regions, 'fig', display_regions_as_points_fig, 'distance_threshold', distance_threshold, 'prefix', sprintf('%s: ACF', basename));
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
        cnn_cache_file = fullfile(cache_dir, 'cnn-cache', sprintf('%s-scale_%g-acf_nms_%g.mat', basename, rescale_image, self.acf_nms_overlap));
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
        self.visualize_detections_as_boxes(I, poly, annotations, detections, 'fig', display_detections_fig, 'multiple_matches', false, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: Final', basename));
    end
    
    % Display detection points
    if display_detections_as_points,
        self.visualize_detections_as_points(I, poly, annotations_pts, detections, 'fig', display_detections_as_points_fig, 'distance_threshold', distance_threshold, 'prefix', sprintf('%s: Final', basename));
    end
end