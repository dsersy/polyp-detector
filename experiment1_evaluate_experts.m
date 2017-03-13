function experiment1_evaluate_experts (varargin)
    % EXPERIMENT1_EVALUATE_EXPERTS (varargin)
    %
    % Evaluates the experts' annotations against the ones made by Martin,
    % in order to obtain human baseline for Experiment 1. The results are
    % displayed on standard output in form of a tab-separated table.
    %
    % Input: optional key/value pairs
    %  - images_list: optional cell array of image filenames (default: {};
    %    use first seven images from Martin's dataset)
    
    % Parse arguments
    parser = inputParser();
    parser.addParameter('images_list', {}, @iscell);
    parser.parse(varargin{:});
    
    images_list = parser.Results.images_list;
    
    % Use first seven images from Martin's dataset by default
    if isempty(images_list)
        dataset_dir = 'dataset-martin';
        images_list = { '01.01.jpg', '02.02.jpg', '02.04.jpg', '05.01.jpg', '07.03.jpg', '100315_TMD_007.jpg', '100315_TMD_022.jpg' };
        images_list = cellfun(@(x) fullfile(dataset_dir, x), images_list, 'UniformOutput', false);
    end
    
    for i = 1:numel(images_list)
        image_file = images_list{i};
        
        %[ ~, image_name ] = fileparts(image_file);
        %fprintf('*** Image #%d/%d: %s ***\n', i, numel(images_list), image_name);
        
        % Get data for the image
        [ I, experiment_basename, polygon, boxes, expert_annotations ] = vicos.PolypDetector.load_data(image_file);
        if isempty(expert_annotations)
            warning('No manual annotations found for the image file: %s!', image_file);
            continue;
        end
        
         % Compute the average polyp dimensions
        box_width = boxes(:,3);
        box_height = boxes(:,4);
        box_diag = sqrt(box_width.^2 + box_height.^2);
        box_diag(box_width == 0 | box_height == 0) = []; % Remove the ones with an invalid dimension
        
        box_centers = boxes(:,1:2) + boxes(:,3:4)/2;
                
        % Distance threshold is based on median size of annotations' diagonals
        distance_threshold = median(box_diag); 
        
        % Validity mask for evaluation (filter out points outside the ROI)
        mask = poly2mask(polygon(:,1), polygon(:,2), size(I, 1), size(I,2));
        
        % Evaluate each expert
        for m = 1:size(expert_annotations, 1)
            expert_name = expert_annotations{m, 1};
            expert_pts  = expert_annotations{m, 2};
            
            % Evaluate
            [ gt, dt ] = vicos.PolypDetector.evaluate_detections_as_points(expert_pts, box_centers, 'validity_mask', mask, 'threshold', distance_threshold);
        
            tp  = sum( gt(:,end) > 0);
            fn  = sum( gt(:,end) == 0);
            tp2 = sum( dt(:,end) > 0);
            fp  = sum( dt(:,end) == 0);
    
            assert(tp == tp2, 'Sanity check failed!');
        
            num_annotations = tp + fn;
            expert_precision = tp / (tp + fp);
            expert_recall = tp / (tp + fn);
            expert_number = tp + fp;
            expert_f_score = 2*(expert_precision*expert_recall)/(expert_precision + expert_recall);
        
            % Display header
            if m == 1
                % Print header
                table_line = strjoin({'%s (%.0f px)', 'Number', 'Ratio', 'Precision', 'Recall', 'F-score \n'}, '\t');
                fprintf(table_line, experiment_basename, distance_threshold);

                % Print ground truth
                table_line = strjoin({'%s', '%d', '', '', '\n' }, '\t');
                fprintf(table_line, 'ground-truth', num_annotations); %#ok<CTPCT>
            end
            
            % Display output            
            table_line = strjoin({'%s', '%d', '%.2f %%', '%.2f %%', '%.2f %%', '%.2f %%\n' }, '\t');
            fprintf(table_line, expert_name, expert_number, 100*expert_number/num_annotations, 100*expert_precision, 100*expert_recall, 100*expert_f_score);
        end
        
        fprintf('\n\n');
    end
end

