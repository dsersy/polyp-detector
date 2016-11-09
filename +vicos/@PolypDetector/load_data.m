function [ I, basename, poly, boxes, manual_annotations ] = load_data (image_filename, varargin)
    % [ I, basename, poly, boxes, manual_annotations ] = LOAD_DATA (image_filename)
    %
    % Loads an image and, if available, its accompanying polygon,
    % bounding box, and point-wise annotations.
    %
    % Input:
    %  - image_filename: input image filename
    %  - varargin: optional key/value pairs
    %     - filter_boxes: filter out boxes that have width or height equal
    %       to zero (default: true)
    %
    % Output:
    %  - I: loaded image
    %  - basename: image's basename, which can be passed to
    %    subsequent processing functions for data caching
    %  - poly: polygon that describes ROI (Nx2 vector of points)
    %  - boxes: manually-annotated bounding boxes (Mx4 matrix, with each
    %    row specifying a bounding box: [ x, y, w, h ]). Any annotations
    %    with width or height equal zero are filterd out.
    %  - manual_annotations: a cell array of point-wise manual
    %    annotations (each row of cell array contains a string denoting the
    %    annotator's name, and a Px2 vector of annotated points)
    
    % Input arguments
    parser = inputParser();
    parser.addParameter('filter_boxes', true, @islogical);
    parser.parse(varargin{:});
    
    filter_boxes = parser.Results.filter_boxes;
    
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
            poly = zeros(0, 2);
        end
    end
    
    % Load annotations (boxes)
    if nargout > 3,
        boxes_file = fullfile(pathname, [ basename, '.bbox' ]);
        if exist(boxes_file, 'file'),
            boxes = load(boxes_file);
            
            % Filter out invalid boxes
            if filter_boxes,
                invalid_mask = any(boxes(:,3:4) == 0, 2);
                boxes(invalid_mask,:) = [];
            end
        else
            boxes = zeros(0, 4);
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