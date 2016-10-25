function fig = visualize_detections_as_boxes (I, polygon, annotations, detections, varargin)
    % fig = VISUALIZE_DETECTIONS_AS_BOXES (I, polygon, annotations, detections, varargin)
    %
    % Visualizes detection/region-proposal bounding boxes.

    parser = inputParser();
    parser.addParameter('fig', [], @(x) isempty(x) || ishandle(x));
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

    % Evaluate detections (if annotations are available)
    if ~isempty(annotations),
        [ gt, det ] = vicos.PolypDetector.evaluate_detections(detections, annotations, 'threshold', overlap_threshold, 'multiple', multiple_matches, 'validity_mask', mask);
    end
    
    % Figure
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

    if ~isempty(annotations),
        % Draw ground-truth; TP and FN
        vicos.utils.draw_boxes(gt(gt(:,5) == 1,:), 'color', 'cyan', 'line_style', '-'); % TP
        vicos.utils.draw_boxes(gt(gt(:,5) == 0,:), 'color', 'yellow', 'line_style', '-'); % FN
        vicos.utils.draw_boxes(gt(gt(:,5) == -1,:), 'color', 'magenta', 'line_style', '-'); % ignore

        % Draw detections; TP and FP
        vicos.utils.draw_boxes(det(det(:,6) == 1,:), 'color', 'green', 'line_style', '-'); % TP
        vicos.utils.draw_boxes(det(det(:,6) == 0,:), 'color', 'red', 'line_style', '-'); % FP

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

        % Set title
        if ~isempty(prefix),
            prefix = sprintf('%s: ', prefix);
        end
        title = sprintf('%srecall: %.2f%%, precision: %.2f%%; counted: %d, annotated: %d ', prefix, recall, precision, num_detected, num_annotated);
    else
        vicos.utils.draw_boxes(detections, 'color', 'green', 'line_style', '-'); % TP
        title = sprintf('%s: num detected: %d', prefix, size(detections, 1));
    end
    
    set(fig, 'Name', title);
    
    % Display as text as well
    h = text(0, 0, title, 'Color', 'white', 'FontSize', 20, 'Interpreter', 'none');
    h.Position(1) = size(I, 2)/2 - h.Extent(3)/2;
    h.Position(2) = h.Extent(4);
        
    % Draw
    drawnow();
end