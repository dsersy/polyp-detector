function [ gt, dt ] = evaluate_detections_as_points (detections, annotations, varargin)
    parser = inputParser();
    parser.addParameter('threshold', 10, @isnumeric);
    parser.parse(varargin{:});
    
    threshold = parser.Results.threshold;
    
    % Detections
    if size(detections, 2) >= 4,
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
    
    for i = 1:numel(assignment),
        if size(dt, 1) <= size(gt, 1),
            dt_idx = i;
            gt_idx = assignment(i);
        else
            gt_idx = i;
            dt_idx = assignment(i);
        end
        
        % LAPJV assigns the invalid (Inf) entries as well, so we need to
        % check fo those manually.
        if isfinite(D(dt_idx, gt_idx)),
            dt(i,end) = gt_idx;
            gt(gt_idx,end) = dt_idx;
        end
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