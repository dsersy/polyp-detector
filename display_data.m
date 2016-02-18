function display_data (image_file, varargin)
    parser = inputParser();
    parser.addParameter('enlarge_box', 0, @isnumeric);
    parser.parse(varargin{:});
    
    enlarge_box = parser.Results.enlarge_box;

    [ path, basename, ~ ] = fileparts(image_file);
    
    I = imread(image_file);
    
    bbox_file = fullfile(path, [ basename, '.bbox' ]);
    poly_file = fullfile(path, [ basename, '.poly' ]);
    
    bbox = load(bbox_file);
    poly = load(poly_file);
    
    % Image
    figure('Name', basename);
    imshow(I);
    hold on;
    
    % Polygon
    polygon = [ poly; poly(1,:) ];
    plot(polygon(:,1), polygon(:,2), 'y-', 'LineWidth', 2);
    
    % Enlarge box
    extra_width  = bbox(:,3)' * enlarge_box;
    extra_height = bbox(:,4)' * enlarge_box;
    
    % Annotations
    x1 = bbox(:,1)' - extra_width/2;
    x2 = bbox(:,1)' + bbox(:,3)' + extra_width;
    y1 = bbox(:,2)' - extra_height/2;
    y2 = bbox(:,2)' + bbox(:,4)' + extra_height;
    line([ x1, x1, x1, x2;
           x2, x2, x1, x2 ], ...
         [ y1, y2, y1, y1;
           y1, y2, y2, y2 ], 'Color', 'red'); 
end