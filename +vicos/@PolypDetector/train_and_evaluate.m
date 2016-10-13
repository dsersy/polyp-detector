function train_and_evaluate (self, result_dir, varargin)
    % TRAIN_AND_EVALUATE (self, result_dir, varargin)
    %
    % Input:
    %  - self:
    %  - result_dir:
    %  - varargin: optional key/value pairs
    %     - train_images: cell array of train image filenames
    %     - test_images: cell array of test image filenames
    %     - display_svm_samples: boolean indicating whether to
    %       visualize SVM training samples or not
    
    parser = inputParser();
    parser.addParameter('train_images', self.default_train_images, @iscell);
    parser.addParameter('test_images', self.default_test_images, @iscell);
    parser.addParameter('display_svm_samples', false, @islogical);
    parser.parse(varargin{:});
    
    train_images = parser.Results.train_images;
    test_images = parser.Results.test_images;
    display_svm_samples = parser.Results.display_svm_samples;
    
    % Cache
    cache_dir = fullfile(result_dir, 'cache');
    
    %% Train SVM
    classifier_file = fullfile(result_dir, 'classifier.mat');
    if exist(classifier_file, 'file'),
        % Load from file
        tmp = load(classifier_file);
        self.svm_classifier = tmp.classifier;
    else
        % Train
        t = tic();
        self.svm_classifier = self.train_svm_classifier('train_images', train_images, 'cache_dir', cache_dir, 'display_svm_samples', display_svm_samples);
        time = toc(t);
        
        % Save
        vicos.utils.ensure_path_exists(classifier_file);
        
        tmp = struct(...
            'classifier', self.svm_classifier, ...
            'time', time); %#ok<NASGU>
        
        save(classifier_file, '-v7.3', '-struct', 'tmp');
    end
    
    %% Test SVM
    all_results = repmat(struct(...
        'image_name', '', ...
        'tp', 0, ...
        'fn', 0, ...
        'fp', 0, ...
        'precision', 0, ...
        'recall', 0, ...
        'num_annotated', 0, ...
        'num_detected', 0), 1, numel(test_images));
    
    for i = 1:numel(test_images),
        test_image = test_images{i};
        
        fprintf('Processing test image #%d/%d: %s\n', i, numel(test_images), test_image);
        
        %% Load test image
        [ I, basename, poly, annotations ] = self.load_data(test_image);
        
        % Try loading result from cache
        results_file = fullfile(result_dir, [ basename, '.mat' ]);
        if exist(results_file, 'file'),
            all_results(i) = load(results_file);
            continue;
        end
        
        % Process
        detections = self.process_image(test_image, 'cache_dir', cache_dir);
        
        %% Evaluate
        % Create mask
        mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
        
        % Evaluate detections
        [ gt, det ] = self.evaluate_detections(detections, annotations, 'threshold', self.evaluation_overlap, 'multiple', false, 'validity_mask', mask);
        
        % Count
        tp = sum( gt(:,5) == 1 );
        fn = sum( gt(:,5) == 0 );
        %tp = sum( det(:,6) == 1 );
        fp = sum( det(:,6) == 0 );
        
        precision = 100*tp/(tp+fp);
        recall = 100*tp/(tp+fn);
        
        num_annotated = sum(gt(:,5) ~= -1);
        num_detected = sum(det(:,6) ~= -1);
        
        %% Store results
        results.image_name = test_image;
        results.tp = tp;
        results.fn = fn;
        results.fp = fp;
        results.precision = precision;
        results.recall = recall;
        results.num_annotated = num_annotated;
        results.num_detected = num_detected;
        
        vicos.utils.ensure_path_exists(results_file);
        save(results_file, '-v7.3', '-struct', 'results');
        
        all_results(i) = results;
    end
    
    save(fullfile(result_dir, 'all_results.mat'), 'all_results');
    
    % Display results
    fprintf('\n\n');
    fprintf('IMAGE_NAME\tREC\tPREC\tRELATIVE\n');
    for i = 1:numel(all_results),
        fprintf('%s\t%3.2f\t%3.2f\t%3.2f\n', all_results(i).image_name, all_results(i).recall, all_results(i).precision, 100*all_results(i).num_detected/all_results(i).num_annotated);
    end
    fprintf('\n');
    fprintf('%s\t%3.2f\t%3.2f\t%3.2f\n', 'AVERAGE', mean([all_results.recall]), mean([all_results.precision]), 100*mean([all_results.num_detected]./[all_results.num_annotated]));
    fprintf('\n');
end