function experiment1_leave_one_out (varargin)
    % EXPERIMENT1_LEAVE_ONE_OUT (varargin)
    %
    % Performs leave-one-out part evaluation of Experiment 1 in the paper.
    % It takes the first seven images from Martin's dataset (v2), and for
    % each image, trains both an ACF detector and CNN/SVM classifier with
    % all other images, then tests on the current image.
    %
    % Input: optional key/value pairs
    %  - output_dir: output directory (default: experiment1-leave-one-out)
    %  - images_list: list of images to use in the experiment (default:
    %    first seven images from Martin's dataset v2)
    %  - negative_folders: list of folders with negative images to use
    %    when training the ACF detector (default: use the cropped negatives
    %    from Kristjan's dataset)
    %  - mix_negatives_with_positives: when training an ACF detector, put
    %    the negative images into folder with labelled images to perform
    %    hard negative mining on all images (default: true)
    %  - acf_window_size: base window size for ACF detector (default: 30
    %    pixels)
    %  - visualize_proposals: create .fig file with visualization of ACF
    %    proposals vs annotations (default: false)
    %  - visualize_detections: create .fig file with visualization of final
    %    detections vs annotations (default: false)
    
    % Input parameters
    parser = inputParser();
    parser.addParameter('output_dir', '', @ischar);
    parser.addParameter('images_list', {}, @iscell);
    parser.addParameter('negative_folders', {}, @iscell);
    parser.addParameter('mix_negatives_with_positives', true, @islogical);
    parser.addParameter('acf_window_size', 30, @isnumeric);
    parser.addParameter('visualize_proposals', false, @islogical);
    parser.addParameter('visualize_detections', false, @islogical);
    parser.addParameter('enhance_images', false, @islogical);
    parser.parse(varargin{:});
    
    output_dir = parser.Results.output_dir;
    images_list = parser.Results.images_list;
    negative_folders = parser.Results.negative_folders;
    mix_negatives_with_positives = parser.Results.mix_negatives_with_positives;
    acf_window_size = parser.Results.acf_window_size;
    visualize_proposals = parser.Results.visualize_proposals;
    visualize_detections = parser.Results.visualize_detections;
    enhance_images = parser.Results.enhance_images;
    
    % Shut the warning about image magnification
    warning('off', 'images:initSize:adjustingMag');

    % Output directory
    if isempty(output_dir),
        output_dir = 'experiment1-leave-one-out';
        
        if enhance_images,
            output_dir = [ output_dir, '_enhance' ];
        end 
    end
    
    % Training images: the first seven images from Martin's dataset (v2)
    if isempty(images_list),
        dataset_dir = 'dataset-martin2';
        images_list = { '01.01.jpg', '02.02.jpg', '02.04.jpg', '05.01.jpg', '07.03.jpg', '100315_TMD_007.jpg', '100315_TMD_022.jpg' };
        images_list = cellfun(@(x) fullfile(dataset_dir, x), images_list, 'UniformOutput', false);
    end
    
    % Additional cropped negatives from Kristjan's dataset
    if isempty(negative_folders),
        negative_folders = { 'dataset-kristjan/negatives-selected' };
    end
    
    % Create a polyp detector pipeline
    polyp_detector = vicos.PolypDetector();
    polyp_detector.enhance_image = enhance_images; % Set globally, so that it is applied at all processing steps
    
    % Pre-allocate results structure
    results = struct(...
        'image_name', '', ...
        'num_annotations', nan, ...
        'distance_threshold', nan, ...
        'proposal_precision', nan, ...
        'proposal_recall', nan, ...
        'proposal_number', nan, ...
        'detection_precision', nan, ...
        'detection_recall', nan, ...
        'detection_number', 0);
    results = repmat(results, 1, numel(images_list));
    
    % Leave-one-out
    for i = 1:numel(images_list),
        test_image = images_list{i};
        
        % Load data
        [ I, experiment_basename, polygon, annotations ] = vicos.PolypDetector.load_data(test_image);
        
        fprintf('*** Image #%d/%d: %s ***\n', i, numel(images_list), test_image);
        
        % Remove the current image from training images
        training_images = images_list;
        training_images(i) = [];
        
        % Cache directory
        cache_dir = fullfile(output_dir, experiment_basename, 'cache');
        
        %% Phase 1: train an ACF detector
        acf_detector_file = fullfile(output_dir, experiment_basename, 'acf_detector.mat');
        if ~exist(acf_detector_file, 'file'),
            % Prepare training dataset
            acf_training_dataset_dir = fullfile(output_dir, experiment_basename, 'acf_training_dataset');
            if ~exist(acf_training_dataset_dir, 'dir'),
                fprintf('Preparing ACF training dataset...\n');
                vicos.AcfDetector.training_prepare_dataset(training_images, acf_training_dataset_dir, 'negative_folders', negative_folders, 'mix_negatives_with_positives', mix_negatives_with_positives, 'enhance_images', enhance_images);
            else
                fprintf('ACF training dataset already exists!\n');
            end
        
            % Train ACF detector
            fprintf('Training ACF detector...\n');
            vicos.AcfDetector.training_train_detector(acf_training_dataset_dir, 'window_size', [ acf_window_size, acf_window_size ], 'output_file', acf_detector_file);
        else
            fprintf('ACF detector has already been trained!\n');
        end
        
        % Set/load the ACF detector
        polyp_detector.load_acf_detector(acf_detector_file);
        
        %% Phase 2: train an SVM classifier
        classifier_file = fullfile(output_dir, experiment_basename, sprintf('classifier-%s.mat', polyp_detector.construct_classifier_identifier()));
        if ~exist(classifier_file, 'file'),
            fprintf('Training SVM classifier...\n');
            t = tic();
            classifier = polyp_detector.train_svm_classifier('train_images', training_images, 'cache_dir', cache_dir); %#ok<NASGU>
            training_time = toc(t); %#ok<NASGU>
            save(classifier_file, '-v7.3', 'classifier', 'training_time');
        else
            fprintf('SVM classifier has already been trained!\n');
        end
        
        % Set/load the classifier
        polyp_detector.load_classifier(classifier_file);
        
        %% Phase 3: test on the current image
        % Compute the average polyp dimensions
        box_width = annotations(:,3);
        box_height = annotations(:,4);
        box_diag = sqrt(box_width.^2 + box_height.^2);
        box_diag(box_width == 0 | box_height == 0) = []; % Remove the ones with an invalid dimension
        
        fprintf(' >> Polyp dimensions: max diagonal: %f, min diagonal: %f, avg diagonal: %f, median diagonal: %f\n', max(box_diag), min(box_diag), mean(box_diag), median(box_diag));
        
        % Distance threshold is based on median size of annotations' diagonals
        distance_threshold = median(box_diag); 
        
        % Validity mask for evaluation (filter out points outside the ROI)
        mask = poly2mask(polygon(:,1), polygon(:,2), size(I, 1), size(I,2));
                 
        % First, get only proposals regions
        regions = polyp_detector.process_image(test_image, 'regions_only', true, 'cache_dir', cache_dir);
        
        % Full detection pipeline
        detections = polyp_detector.process_image(test_image, 'cache_dir', cache_dir);
        
        fprintf(' >> %d regions, %d detections; %d annotations\n', size(regions, 1), size(detections, 1), size(annotations, 1));
        
        %% Evaluation
        % Point-wise evaluation
        box_centers = annotations(:,1:2) + annotations(:,3:4)/2;
        
        % Evaluate proposals
        [ gt, dt ] = vicos.PolypDetector.evaluate_detections_as_points(regions, box_centers, 'validity_mask', mask, 'threshold', distance_threshold);
        
        tp  = sum( gt(:,end) > 0);
        fn  = sum( gt(:,end) == 0);
        tp2 = sum( dt(:,end) > 0);
        fp  = sum( dt(:,end) == 0);
        
        assert(tp == tp2, 'Sanity check failed!');
        
        num_annotations = tp + fn;
        proposal_precision = tp / (tp + fp);
        proposal_recall = tp / (tp + fn);
        proposal_number = tp + fp;
        
        fprintf('proposals; precision: %.2f %%, recall: %.2f %%, number detected: %d, number annotated: %d, ratio: %.2f %%\n', 100*proposal_precision, 100*proposal_recall, proposal_number, num_annotations, 100*proposal_number/num_annotations);
        
        % Evaluate final detections
        [ gt, dt ] = vicos.PolypDetector.evaluate_detections_as_points(detections, box_centers, 'validity_mask', mask, 'threshold', distance_threshold);
        
        tp  = sum( gt(:,end) > 0);
        fn  = sum( gt(:,end) == 0);
        tp2 = sum( dt(:,end) > 0);
        fp  = sum( dt(:,end) == 0);
        
        assert(tp == tp2, 'Sanity check failed!');
        
        num_annotations = tp + fn;
        detection_precision = tp / (tp + fp);
        detection_recall = tp / (tp + fn);
        detection_number = tp + fp;
        
        fprintf('detections; precision: %.2f %%, recall: %.2f %%, number detected: %d, number annotated: %d, ratio: %.2f %%\n', 100*detection_precision, 100*detection_recall, detection_number, num_annotations, 100*detection_number/num_annotations);
        
        % Store results
        results(i).image_name = experiment_basename;
        results(i).num_annotations = num_annotations;
        results(i).distance_threshold = distance_threshold;
        results(i).proposal_precision = proposal_precision;
        results(i).proposal_recall = proposal_recall;
        results(i).proposal_number = proposal_number;
        results(i).detection_precision = detection_precision;
        results(i).detection_recall = detection_recall;
        results(i).detection_number = detection_number;
        
        %% Visualization (optional)
        if visualize_proposals,
            fig = figure('Visible', 'off');
            vicos.PolypDetector.visualize_detections_as_points(I, polygon, box_centers, regions, 'fig', fig, 'distance_threshold', distance_threshold, 'prefix', sprintf('%s: ACF proposals', experiment_basename));
            savefig(fig, fullfile(output_dir, sprintf('%s-proposals.fig', experiment_basename)), 'compact');
            delete(fig);
        end
        
        if visualize_detections,
            fig = figure('Visible', 'off');
            vicos.PolypDetector.visualize_detections_as_points(I, polygon, box_centers, detections, 'fig', fig, 'distance_threshold', distance_threshold, 'prefix', sprintf('%s: final detections', experiment_basename));
            savefig(fig, fullfile(output_dir, sprintf('%s-proposals.fig', experiment_basename)), 'compact');
            delete(fig);
        end
    end
    
    %% Display results again (for copy & paste purposes)
    fprintf('\n\n');
    table_line = strjoin({'Image name', 'Num annotations', 'Distance threshold', 'Proposal precision', 'Proposal recall', 'Num proposals', 'Proposal ratio', 'Detection precision', 'Detection recall', 'Num detections', 'Detection ratio\n'}, '\t');
    fprintf(table_line);
    for i = 1:numel(results),
        table_line = strjoin({'%s', '%d', '%.0f px', '%.2f %%', '%.2f %%', '%d', '%.2f %%', '%.2f %%', '%.2f %%', '%d', '%.2f %%\n'}, '\t');
        fprintf(table_line, results(i).image_name, results(i).num_annotations, results(i).distance_threshold, ...
            100*results(i).proposal_precision, 100*results(i).proposal_recall, results(i).proposal_number, 100*results(i).proposal_number/results(i).num_annotations, ...
            100*results(i).detection_precision, 100*results(i).detection_recall, results(i).detection_number, 100*results(i).detection_number/results(i).num_annotations);
    end
    fprintf('\n\n');
end