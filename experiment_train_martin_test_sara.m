function experiment_train_martin_test_sara (training_dataset, detector)
    if ~exist('detector', 'var')
        detector = [];
    end
    
    distance_threshold = 32;
    
    % Training dataset selection
    switch training_dataset
        case 'martin'
            training_images = {
                'dataset-martin/01.01.jpg', ...
                'dataset-martin/02.02.jpg', ...
                'dataset-martin/02.04.jpg', ...
                'dataset-martin/05.01.jpg', ...
                'dataset-martin/07.03.jpg', ...
                'dataset-martin/100315_TMD_007.jpg', ...
                'dataset-martin/100315_TMD_022.jpg' };

            results_dir = 'results-martin-sara';
        case 'martin2'
            training_images = {
                'dataset-martin2/01.01.jpg', ...
                'dataset-martin2/02.02.jpg', ...
                'dataset-martin2/02.04.jpg', ...
                'dataset-martin2/05.01.jpg', ...
                'dataset-martin2/07.03.jpg', ...
                'dataset-martin2/100315_TMD_007.jpg', ...
                'dataset-martin2/100315_TMD_022.jpg', ...
                'dataset-martin2/120413_TMD_011.jpg', ...
                'dataset-martin2/120413_TMD_022.jpg', ...
                'dataset-martin2/120914_TJASAp_017.jpg', ...
                'dataset-martin2/120914_TJASAp_021.jpg', ...
                'dataset-martin2/121113_TMD_003.jpg', ...
                'dataset-martin2/121113_TMD_018.jpg' };
            
            results_dir = 'results-martin2-sara';    
        otherwise
            error('Invalid training dataset: %s!', training_dataset);
    end
    
    cache_dir = fullfile(results_dir, 'cache');
    
    % Create polyp detector
    if isempty(detector)
        detector = vicos.PolypDetector();
    end
    
    detector.svm_classifier = []; % Reset classifier, just to be sure
    
    %% Train classifier
    classifier_identifier = detector.construct_classifier_identifier();
    
    classifier_file = fullfile(results_dir, sprintf('classifier-%s.mat', classifier_identifier));
    if ~exist(classifier_file, 'file')
        detector.train_svm_classifier('cache_dir', cache_dir, 'train_images', training_images);
        classifier = detector.svm_classifier; %#ok<NASGU>
        save(classifier_file, '-v7.3', 'classifier');
    else
        detector.load_classifier(classifier_file);
    end
    
    %% Test classifier
    dataset_dir = 'dataset-sara';
    test_images = dir(fullfile(dataset_dir, '*.jpg'));
    test_images = arrayfun(@(x) fullfile(dataset_dir, x.name), test_images, 'UniformOutput', false);

    % Process all images
    all_results = cell(numel(test_images), 1);
    for i = 1:numel(test_images)
        [ I, basename, polygon, ~, manual_annotations ] = detector.load_data(test_images{i});

        results_file = fullfile(results_dir, classifier_identifier, [ basename, '.mat' ]);
            
        if ~exist(results_file, 'file')
            % Get annotations
            annotations = manual_annotations{2};
            
            % Process image; first, detections only
            regions = detector.process_image(test_images{i}, 'cache_dir', cache_dir, 'regions_only', true);
            
            % Process image; second, whole pipeline
            detections = detector.process_image(test_images{i}, 'cache_dir', cache_dir, 'regions_only', false);
            
            % Save
            results = struct('image_name', test_images{i}, ...
                             'image_size', size(I), ...
                             'base_name', basename, ...
                             'polygon', polygon, ...
                             'annotations', annotations, ...
                             'regions', regions, ...
                             'detections', detections);
            
            vicos.utils.ensure_path_exists(results_file);
            save(results_file, '-v7.3', '-struct', 'results');
        else
            results = load(results_file);
        end
        
        all_results{i} = results;
    end

    
    %% Display results       
    fprintf('image name\tresolution\tnum annotated\tnum regions\tprecision\trecall\tportion\tnumdetections\tprecision\trecall\tportion\n');
    for i = 1:numel(all_results)
        results = all_results{i};
        
        resolution_string = sprintf('%dx%d', results.image_size(2), results.image_size(1));
        
        num_annotations = size(results.annotations, 1);
        
        regions_num = size(results.regions, 1);
        
        % Evaluate regions
        [ evaluation_regions.gt, evaluation_regions.dt ] = detector.evaluate_detections_as_points(results.regions, results.annotations, 'threshold', distance_threshold);    
        [ regions_precision, regions_recall ] = compute_precision_recall(evaluation_regions);
        regions_portion = regions_num/num_annotations;
        
        % Evaluate detections
        [ evaluation_detections.gt, evaluation_detections.dt ] = detector.evaluate_detections_as_points(results.detections, results.annotations, 'threshold', distance_threshold);
        detections_num = size(results.detections, 1);
        [ detections_precision, detections_recall ] = compute_precision_recall(evaluation_detections);
        detections_portion = detections_num / num_annotations;
        
        format = strjoin({ '%s', '%s', '%d', '%d', '%.2f %%', '%.2f %%', '%.2f %%', '%d', '%.2f %%', '%.2f %%', '%.2f %%', '\n' }, '\t');
        fprintf(format, ...
            results.base_name, ...
            resolution_string, ...
            num_annotations, ...
            regions_num, regions_precision*100, regions_recall*100, regions_portion*100, ...
            detections_num, detections_precision*100, detections_recall*100, detections_portion*100);
        
    end
end

function [ precision, recall ] = compute_precision_recall (evaluation)
    tp = sum(evaluation.dt(:,end) >  0);
    fp = sum(evaluation.dt(:,end) == 0);
        
    %tp = sum(evaluation.gt(:,end) >  0); % Should be equal to the one we got from detections
    fn = sum(evaluation.gt(:,end) == 0);
        
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
end
