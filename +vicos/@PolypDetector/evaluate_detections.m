function [ gt, dt ] = evaluate_detections (detections, annotations, varargin)
    % [ gt, dt ] = EVALUATE_DETECTIONS (detections, annotations, varargin)
    %
    % Evaluates detections against provided annotations, using bounding box
    % overlap.
    %
    % Input:
    %  - detections: Nx4 array of detection bounding boxes ([ x, y, w, h ]). 
    %    The matrix may contain additional columns (e.g., detection 
    %    scores), which will be ignored.
    %  - annotations: Mx4 array of annotation bounding boxes ([ x, y, w, h ]). 
    %    The matrix is internally augmented with fifth column representing
    %    ignore flag.
    %  - varargin: optional key/value pairs:
    %     - threshold: minimum overlap threshold (default: 0.5)
    %     - multiple: allow multiple detection assignments (default: false)
    %     - validity_mask: optional boolean validity mask, used to
    %       determine which annotations should be marked for ignoring. If
    %       provided, it must be large enough to accomodate coordinates of
    %       all given annotations.
    %
    % Output:
    %  - gt: Mx5 matrix, where last column denotes the assignment status:
    %     0 = unassigned, false negative
    %     1 = assigned, true positive
    %    -1 = ignore
    %  - dt: Nx5 matrix (or Nx(D+1) if detections contained more than four
    %    columns), where last column denotes the assignment status:
    %     0 = unassigned, false positive
    %     1 = assigned, true positive
    %    -1 = ignore
    %
    % Note: this function makes use of Piotr Dollar's bbGt function in
    % 'evalRes' mode to perform evaluation
    
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