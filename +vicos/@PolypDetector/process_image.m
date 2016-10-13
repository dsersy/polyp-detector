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
        self.visualize_detections_or_regions(I, poly, annotations, regions, 'multiple_matches', true, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: ACF', basename));
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
        self.visualize_detections_or_regions(I, poly, annotations, detections, 'multiple_matches', false, 'overlap_threshold', overlap_threshold, 'prefix', sprintf('%s: Final', basename));
    end
    
    % Display detection points
    if display_detections_as_points,
        self.visualize_detections_as_points(I, poly, annotations_pts, detections, 'prefix', sprintf('%s: Final points', basename));
    end
end