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
    %      (default: use the object's evaluation_distance setting)
    %    - enhance_image: logical flag indicating whether image should be
    %      enchanced with Contrast-Limited Adaptive Histogram Equalization
    %      before being processed (default: use the object's enhance_image
    %      setting)
    %    - rescale_image: scale factor by which the image should be resized
    %      before processing. Note that rescaling is applied after image is
    %      cropped to the ROI size.
    %    - display_timings: print timings for individual stages (default:
    %      false)
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
    parser.addParameter('enhance_image', self.enhance_image, @islogical);
    parser.addParameter('rescale_image', 1.0, @isnumeric);
    parser.addParameter('display_timings', false, @islogical);
    parser.parse(varargin{:});
    
    cache_dir = parser.Results.cache_dir;
    regions_only = parser.Results.regions_only;
    display_regions = parser.Results.display_regions;
    display_regions_as_points = parser.Results.display_regions_as_points;
    display_detections = parser.Results.display_detections;
    display_detections_as_points = parser.Results.display_detections_as_points;
    overlap_threshold = parser.Results.overlap_threshold;
    distance_threshold = parser.Results.distance_threshold;
    enhance_image = parser.Results.enhance_image;
    rescale_image = parser.Results.rescale_image;
    
    display_timings = parser.Results.display_timings;
    
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
    [ Iorig, basename, poly, annotations, annotations_pts ] = self.load_data(image_filename);
    
    t_total = tic();
    
    % Enhance the image
    if enhance_image,
        I = vicos.utils.adaptive_histogram_equalization(Iorig, 'NumTiles', [ 16, 16 ]);
    else
        I = Iorig;
    end
    
    % Mask the image
    Im = self.mask_image_with_polygon(I, poly);
    
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
    cache_basename = self.construct_cache_filename(basename, enhance_image, rescale_image, self.acf_nms_overlap);
    
    %% Run ACF detector
    if ~isempty(cache_dir),
        acf_cache_file = fullfile(cache_dir, 'acf-cache', [ cache_basename, '.mat' ]);
    else
        acf_cache_file = '';
    end
    
    % Detect
    t = tic();
    
    regions = self.detect_candidate_regions(Im, acf_cache_file);
    
    % Undo the scaling
    regions = regions / rescale_image;
    
    % Undo the effect of the crop
    regions(:,1) = regions(:,1) + xmin;
    regions(:,2) = regions(:,2) + ymin;

    if display_timings,
        fprintf(' > Timings: region proposal: %f seconds\n', toc(t));
    end
    
    % Display ACF regions
    if display_regions,
        self.visualize_detections_as_boxes(Iorig, poly, annotations, regions, 'fig', display_regions_fig, 'multiple_matches', true, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: ACF', basename));
    end
    
    % Display ACF regions as points
    if display_regions_as_points,
        if isempty(annotations_pts) && ~isempty(annotations),
            annotations_pts = { 'Annotated box centers', annotations(:,1:2) + annotations(:,3:4)/2 };
        end
        
        self.visualize_detections_as_points(Iorig, poly, annotations_pts, regions, 'fig', display_regions_as_points_fig, 'distance_threshold', distance_threshold, 'prefix', sprintf('%s: ACF', basename));
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
        cnn_cache_file = fullfile(cache_dir, 'cnn-cache', [ cache_basename, '.mat' ]);
    else
        cnn_cache_file = '';
    end
    
    % Extract
    t = tic();

    features = self.extract_features_from_regions(I, regions, cnn_cache_file);
    
    if display_timings,
        fprintf(' > Timings: CNN feature extraction: %f seconds\n', toc(t));
    end
    
    %% Classify with SVM
    fprintf('Performing SVM classification...\n');
    
    t = tic();
    
    [ ~, scores ] = self.svm_classifier.predict(features);
    
    positive_mask = scores >= 0;
    positive_regions = regions(positive_mask,1:4);
    positive_scores = scores(positive_mask);
    
    if display_timings,
        fprintf(' > Timings: SVM: %f seconds\n', toc(t));
    end
    
    %% Additional NMS on top of SVM predictions
    fprintf('Performing NMS on top of SVM predictions...\n');
    
    t = tic();
    
    detections = bbNms([ positive_regions, positive_scores' ], ...
        'type', 'maxg', ...
        'overlap', self.svm_nms_overlap, ...
        'ovrDnm', 'union');
    
    if display_timings,
        fprintf(' > Timings: NMS: %f seconds\n', toc(t));
        fprintf(' >> Timings: total: %f seconds\n', toc(t_total));
    end

    % Display detections
    if display_detections,
        self.visualize_detections_as_boxes(Iorig, poly, annotations, detections, 'fig', display_detections_fig, 'multiple_matches', false, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: Final', basename));
    end
    
    % Display detection points
    if display_detections_as_points,
        if isempty(annotations_pts) && ~isempty(annotations),
            annotations_pts = { 'Annotated box centers', annotations(:,1:2) + annotations(:,3:4)/2 };
        end
        
        self.visualize_detections_as_points(Iorig, poly, annotations_pts, detections, 'fig', display_detections_as_points_fig, 'distance_threshold', distance_threshold, 'prefix', sprintf('%s: Final', basename));
    end
end