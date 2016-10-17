function fig = visualize_detections_as_points (I, polygon, annotations, detections, varargin)
    % fig = VISUALIZE_DETECTIONS_AS_POINTS (I, polygon, annotations, detections, varargin)
    %
    % Visualizes detection/region proposal centroids.

    parser = inputParser();
    parser.addParameter('fig', [], @ishandle);
    parser.addParameter('prefix', '', @ischar);
    parser.addParameter('threshold', 32, @isnumeric);
    parser.addParameter('evaluate_against', '', @ischar);
    parser.parse(varargin{:});

    fig = parser.Results.fig;
    prefix = parser.Results.prefix;
    threshold = parser.Results.threshold;
    evaluate_against = parser.Results.evaluate_against;

    % Create mask
    mask = poly2mask(polygon(:,1), polygon(:,2), size(I, 1), size(I,2));

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
    
    if ~isempty(evaluate_against) || size(annotations, 1) == 1,
        if size(annotations, 1) == 1,
            % There's only one set of annotations, and we evaluate against
            % it
            idx = 1;
        else
            % Evaluate against selected set of manual annotations
            idx = find(ismember(annotations(:,1), evaluate_against));
        end
        
        name = annotations{idx, 1};
        ground_truth = annotations{idx, 2};
        
        % Evaluate
        [ gt, dt ] = vicos.PolypDetector.evaluate_detections_as_points(detections, ground_truth, 'threshold', threshold);
        
        % Draw ground-truth; TP and FN
        gt_assigned = gt(:,end) ~= 0;
        plot(gt(gt_assigned,1), gt(gt_assigned, 2), '+', 'Color', 'cyan', 'MarkerSize', 8, 'LineWidth', 2); % TP
        plot(gt(~gt_assigned,1), gt(~gt_assigned, 2), '+', 'Color', 'yellow', 'MarkerSize', 8, 'LineWidth', 2); % FN
        
        % Draw detections; TP and FP
        dt_assigned = dt(:,end) ~= 0;
        plot(dt(dt_assigned,1), dt(dt_assigned, 2), 'x', 'Color', 'green', 'MarkerSize', 8, 'LineWidth', 2); % TP
        plot(dt(~dt_assigned,1), dt(~dt_assigned, 2), 'x', 'Color', 'red', 'MarkerSize', 8, 'LineWidth', 2); % FP
        
        % Draw assignments
        for p = 1:size(dt, 1),
            midx = dt(p,end);
            if midx > 0,
                plot([ dt(p,1), gt(midx, 1) ], [ dt(p,2), gt(midx, 2) ], 'c-', 'LineWidth', 1.5);
            end
        end
        for p = 1:size(gt, 1),
            midx = gt(p,end);
            if midx > 0,
                plot([ gt(p,1), dt(midx, 1) ], [ gt(p,2), dt(midx, 2) ], 'g-', 'LineWidth', 1.5, 'LineStyle', '--');
            end
        end

        % Create fake plots for legend entries
        h = [];
        h(end+1) = plot([0,0], [0,0], '+', 'Color', 'cyan', 'LineWidth', 2);
        h(end+1) = plot([0,0], [0,0], '+', 'Color', 'yellow', 'LineWidth', 2);
        h(end+1) = plot([0,0], [0,0], 'x', 'Color', 'green', 'LineWidth', 2);
        h(end+1) = plot([0,0], [0,0], 'x', 'Color', 'red', 'LineWidth', 2);
        legend(h, 'TP (annotated)', 'FN', 'TP (det)', 'FP');
        
        % Numeric evaluation
        tp = sum(dt(:,end)  > 0);
        fp = sum(dt(:,end) == 0);
        %tp = sum(gt(:,end)  > 0);
        fn = sum(gt(:,end) == 0);
        
        precision = tp / (tp + fp)*100;
        recall    = tp / (tp + fn)*100;
        
        num_detected = size(dt, 1);
        num_annotated = size(gt, 1);
        
        % Set title
        if ~isempty(prefix),
            prefix = sprintf('%s: ', prefix);
        end
        title = sprintf('%srecall: %.2f%%, precision: %.2f%%; counted: %d, annotated: %d ', prefix, recall, precision, num_detected, num_annotated);
    else
        % Draw all manual annotations
        h = [];
        legend_entries = {};

        if ~isempty(annotations),
            num_annotations = size(annotations, 1);
            colors = lines(num_annotations);
            for i = 1:num_annotations,
                annotation_id = annotations{i, 1};
                annotation_points = annotations{i, 2};

                h(end+1) = plot(annotation_points(:,1), annotation_points(:,2), 'ko', 'MarkerFaceColor', colors(i,:));
                legend_entries{end+1} = sprintf('%s (%d)', annotation_id, size(annotation_points, 1));
            end
        end

        % Draw all detections
        detection_points = detections(:,1:2) + detections(:,3:4)/2;
        h(end+1) = plot(detection_points(:,1), detection_points(:,2), 'gx', 'MarkerSize', 8, 'LineWidth', 2);
        legend_entries{end+1} = sprintf('Detector (%d)', size(detections, 1));
        
        % Legend
        legend(h, legend_entries, 'Location', 'NorthEast', 'Interpreter', 'none');

        title = prefix;
    end
    
    % Set title    
    set(fig, 'Name', title);
    
    % Display as text as well
    h = text(0, 0, title, 'Color', 'white', 'FontSize', 20, 'Interpreter', 'none');
    h.Position(1) = size(I, 2)/2 - h.Extent(3)/2;
    h.Position(2) = h.Extent(4);
    
    % Draw
    drawnow();
end