function experiment1_leave_one_out (varargin)
    % EXPERIMENT1_LEAVE_ONE_OUT ()
    
    parser = inputParser();
    parser.addParameter('output_dir', 'experiment1-leave-one-out', @ischar);
    parser.addParameter('images_list', {}, @iscell);
    parser.addParameter('negative_folders', {}, @iscell);
    parser.addParameter('mix_negatives_with_positives', true, @islogical);
    parser.addParameter('acf_window_size', 30, @isnumeric);
    parser.parse(varargin{:});
    
    output_dir = parser.Results.output_dir;
    images_list = parser.Results.images_list;
    negative_folders = parser.Results.negative_folders;
    mix_negatives_with_positives = parser.Results.mix_negatives_with_positives;
    acf_window_size = parser.Results.acf_window_size;
    
    % Training images: the first seven images from Martin's dataset (v2)
    if isempty(images_list),
        dataset_dir = 'dataset-martin2';
        images_list = { '01.01.jpg', '02.02.jpg', '02.04.jpg', '05.01.jpg', '07.03.jpg', '100315_TMD_007.jpg', '100315_TMD_022.jpg'};
        images_list = cellfun(@(x) fullfile(dataset_dir, x), images_list, 'UniformOutput', false);
    end
    
    % Additional cropped negatives from Kristjan's dataset
    if isempty(negative_folders),
        negative_folders = { 'dataset-kristjan/negatives-selected' };
    end
    
    % Create a polyp detector
    polyp_detector = vicos.PolypDetector();
    
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
                vicos.AcfDetector.training_prepare_dataset(training_images, acf_training_dataset_dir, 'negative_folders', negative_folders, 'mix_negatives_with_positives', mix_negatives_with_positives);
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
        mask = poly2mask(polygon(:,1), polygon(:,2), size(I, 1), size(I,2));
         
        rescale_image = 1.0;
        enhance_image = false;
         
        % First, get only proposals regions
        fig = figure('Visible', 'off');
        regions = polyp_detector.process_image(test_image, 'regions_only', true, 'cache_dir', cache_dir, 'rescale_image', rescale_image, 'enhance_image', enhance_image, 'display_regions_as_points', fig);
        savefig(fullfile(output_dir, sprintf('%s-reg.fig', experiment_basename)));
        delete(fig);
        
        % Full detection pipeline
        fig = figure('Visible', 'off');
        detections = polyp_detector.process_image(test_image, 'cache_dir', cache_dir, 'rescale_image', rescale_image, 'enhance_image', enhance_image, 'display_detections_as_points', fig);
        savefig(fullfile(output_dir, sprintf('%s-det.fig', experiment_basename)));
        delete(fig);
        
        fprintf(' >> %d regions, %d detections; %d annotations\n', size(regions, 1), size(detections, 1), size(annotations, 1));
        
%         [ gt, dt ] = vicos.PolypDetector.evaluate_detections_as_points(regions, annotations, 'validity_mask', mask, 'threshold', distance_threshold);
%         
%         % Full detection pipeline
%         
%         [ gt, dt ] = vicos.PolypDetector.evaluate_detections_as_points(detections, annotations, 'validity_mask', mask, 'threshold', distance_threshold);
        
    end
end