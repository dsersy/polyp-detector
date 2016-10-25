function leave_one_out_cross_validation (self, result_dir, varargin)
    % LEAVE_ONE_OUT_CROSS_VALIDATION (self, result_dir, varargin)
    
    parser = inputParser();
    parser.addParameter('images', {}, @iscell);
    parser.addParameter('store_visualizations', false, @islogical);
    parser.parse(varargin{:});
    
    images = parser.Results.images;
    if isempty(images),
        images = union(self.default_train_images, self.default_test_images);
    end
    
    store_visualizations = parser.Results.store_visualizations;
    
    if store_visualizations,
        fig_boxes = figure('Visible', 'off');
        fig_points = figure('Visible', 'off');
    end
    
    % Cache
    cache_dir = fullfile(result_dir, 'cache');
    
    % Create list of images
    
    %% Leave-one-out loop
    all_results = repmat(struct(...
        'image_name', '', ...
        'tp', 0, ...
        'fn', 0, ...
        'fp', 0, ...
        'precision', 0, ...
        'recall', 0, ...
        'num_annotated', 0, ...
        'num_detected', 0), 1, numel(images));
    
    for i = 1:numel(images),
        train_images = images;
        train_images(i) = [];
        
        test_image = images{i};
        
        % Load test image
        [ I, basename, poly, annotations ] = self.load_data(test_image);
        
        % Try loading result from cache
        results_file = fullfile(result_dir, [ basename, '.mat' ]);
        if exist(results_file, 'file'),
            all_results(i) = load(results_file);
            continue;
        end
        
        %% Train SVM
        classifier_file = fullfile(result_dir, 'classifiers', [ basename, '.mat' ]);
        if exist(classifier_file, 'file'),
            % Load from file
            tmp = load(classifier_file);
            self.svm_classifier = tmp.classifier;
        else
            % Train
            t = tic();
            self.svm_classifier = self.train_svm_classifier('train_images', train_images, 'cache_dir', cache_dir);
            time = toc(t);
            
            % Save
            vicos.utils.ensure_path_exists(classifier_file);
            
            tmp = struct(...
                'classifier', self.svm_classifier, ...
                'time', time); %#ok<NASGU>
            
            save(classifier_file, '-v7.3', '-struct', 'tmp');
        end
        
        %% Process the left-out image
        if store_visualizations,
            extra_args = { 'display_detections', fig_boxes, 'display_detections_as_points', fig_points };
        else
            extra_args = false;
        end
        
        detections = self.process_image(test_image, 'cache_dir', cache_dir, extra_args{:});
        
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
        
        % Save visualization
        if store_visualizations,
            figure_file = fullfile(result_dir, [ basename, '-boxes.fig' ]);
            savefig(fig_boxes, figure_file);
            
            figure_file = fullfile(result_dir, [ basename, '-points.fig' ]);
            savefig(fig_points, figure_file);
        end
        
        all_results(i) = results;
    end
    
    save(fullfile(result_dir, 'all_results.mat'), '-v7.3', 'all_results');
end