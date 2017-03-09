function [ gt, dt ] = evaluate_detections_as_points (detections, annotations, varargin)
    % [ gt, dt ] = EVALUATE_DETECTIONS_AS_POINTS (detections, annotations, varargin)
    %
    % Evaluates detections against provided annotations, using center point
    % distance.
    %
    % Input:
    %  - detections: Nx2 matrix of detection centroids; it may be NxD; if
    %    D >= 4, it is assumed that the first four columns are specify
    %    bounding boxes ([ x, y, w, h ]), and are used to compute the
    %    centroids.
    %  - annotations: Mx2 matrix of annotation centroids
    %  - varargin: optional key/value pairs
    %     - threshold: distance threshold (default: 10 pixels)
    %     - validity_mask: optional boolean validity mask, used to
    %       determine which annotations should be marked for ignoring. If
    %       provided, it must be large enough to accomodate coordinates of
    %       all given annotations.
    %
    % Output:
    %  - gt: Nx3 (or Nx(D+1)) matrix, where last column represents the
    %    index of assigned detection. If it is 0, it means the annotation
    %    was left unassigned (i.e., false negative). If it is -1, it means
    %    the annotation was ignored.
    %  - dt: Mx3 matrix, where last column represents the index of assigned
    %    annotation. If it is 0, it means the detection was left unassigned
    %    (i.e., false positive). If it is -1, it means the detection was
    %    ignored.
    %
    % Note: this function internally uses Jonker-Volgenant linear
    % assignment algorith to find optimal assignment between the given set
    % of detections and annotations.
    
    parser = inputParser();
    parser.addParameter('threshold', 10, @isnumeric);
    parser.addParameter('validity_mask', [], @islogical);
    parser.parse(varargin{:});
    
    threshold = parser.Results.threshold;
    validity_mask = parser.Results.validity_mask;

    % Detections
    if size(detections, 2) >= 4
        % If detections matrix has four or more columns, assume that we
        % are given detections as boxes (plus optional scores) and convert
        % them to centroids
        dt = detections(:,1:2) + detections(:,3:4)/2;
    else
        dt = detections(:,1:2);
    end
    
    % Annotations
    assert(isnumeric(annotations) && size(annotations, 2) == 2, 'Annotations must be given as Nx2 matrix of centroids!');
    gt = annotations;
    
    % Compute the distance matrix; NxM where N is number of detections and
    % M is number of annotations. In other words, each row corresponds to a
    % detection and each column corresponds to an annotation.
    D = distance_matrix(dt, gt);
    D(D > threshold) = Inf; % Mask invalid entries
    
    assignment = lapjv(D);
    
    % Augment with assignment indices
    dt(:,end+1) = 0;
    gt(:,end+1) = 0;
    
    for i = 1:numel(assignment)
        if size(dt, 1) <= size(gt, 1)
            dt_idx = i;
            gt_idx = assignment(i);
        else
            gt_idx = i;
            dt_idx = assignment(i);
        end
        
        % LAPJV assigns the invalid (Inf) entries as well, so we need to
        % check fo those manually.
        if isfinite(D(dt_idx, gt_idx))
            dt(dt_idx,end) = gt_idx;
            gt(gt_idx,end) = dt_idx;
        end
    end
    
    if ~isempty(validity_mask)
        % Filter ground-truth
        x = round(gt(:,1));
        y = round(gt(:,2));
        ignore_gt_idx = ~validity_mask(sub2ind(size(validity_mask), y, x));
        ignore_dt_idx = gt(ignore_gt_idx, end); % Get list of detections assigned to ignored annotations
        ignore_dt_idx(ignore_dt_idx <= 0) = []; % Remove unassigned or already-ignored
        gt(ignore_gt_idx,end) = -1;
        dt(ignore_dt_idx,end) = -1;
        
        % Filter detections
        x = round(dt(:,1));
        y = round(dt(:,2));
        ignore_dt_idx = ~validity_mask(sub2ind(size(validity_mask), y, x));
        ignore_gt_idx = dt(ignore_dt_idx, end); % Get list of annotations assigned to ignored annotations
        ignore_gt_idx(ignore_gt_idx <= 0) = []; % Remove unassigned or already-ignored
        gt(ignore_gt_idx,end) = -1;
        dt(ignore_dt_idx,end) = -1;
    end
end

function D = distance_matrix (X, Y)
    % D = DISTANCE_MATRIX (X, Y)
    %
    % Computes a matrix of pair-wise Euclidean distances between two sets
    % of points, X and Y.
    %
    % Input:
    %  - X: MxD vector of points
    %  - Y: NxD vector of points
    %
    % Output:
    %  - MxN distance matrix
    
    Yt = Y';
    XX = sum(X .* X, 2);
    YY = sum(Yt .* Yt, 1);
    D = bsxfun(@plus, XX, YY) - 2*X*Yt;
    D = sqrt(D);
end