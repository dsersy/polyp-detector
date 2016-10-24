function svm = train_svm_classifier (self, varargin)
    % svm = TRAIN_SVM_CLASSIFIER (self, varargin)
    %
    % Trains an SVM classifier.
    %
    % Input:
    %  - self:
    %  - varagin: optional key/value pairs:
    %     - cache_dir: cache directory (default: '')
    %     - train_images: cell array of train image file names
    %       (default: use built-in list)
    %     - display_svm_samples: visualize SVM training samples on
    %       each training image
    %
    % Output:
    %  - svm: trained SVM classifier
    
    % Input arguments
    parser = inputParser();
    parser.addParameter('cache_dir', '', @ischar);
    parser.addParameter('train_images', self.default_train_images, @iscell);
    parser.addParameter('display_svm_samples', false, @islogical);
    parser.parse(varargin{:});
    
    cache_dir = parser.Results.cache_dir;
    train_images = parser.Results.train_images;
    display_svm_samples = parser.Results.display_svm_samples;
    
    %% Process all train images to get the features and boxes
    num_images = numel(train_images);
    
    all_features = cell(1, num_images);
    all_labels = cell(1, num_images);
    for i = 1:num_images,
        image_file = train_images{i};
        fprintf('Processing train image #%d/%d: %s\n', i, num_images, train_images{i});
        
        % Detect regions in the image
        [ I, basename, poly, annotations ] = self.load_data(image_file);
        regions = self.process_image(image_file, 'regions_only', true, 'cache_dir', cache_dir);
        
        % Extract CNN features
        % NOTE: we extract features from all regions, even the ones
        % that will be later discarded, in order to keep cache
        % files consistent with those produced by the
        % process_image() function!
        if ~isempty(cache_dir),
            rescale_image = 1; % We do not support rescaling here...
            cnn_cache_file = fullfile(cache_dir, 'cnn-cache', sprintf('%s-scale_%g-acf_nms_%g.mat', basename, rescale_image, self.acf_nms_overlap));
        else
            cnn_cache_file = '';
        end
        features = self.extract_features_from_regions(I, regions, cnn_cache_file);
        
        % Determine whether boxes are positive or negative
        mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
        [ ~, regions_pos ] = self.evaluate_detections(regions, annotations, 'threshold', self.training_positive_overlap, 'multiple', true, 'validity_mask', mask);
        [ ~, regions_neg ] = self.evaluate_detections(regions, annotations, 'threshold', self.training_negative_overlap, 'multiple', true, 'validity_mask', mask);
        
        % Boxes must be strictly positive or negative (this will
        % filter out the ones that were marked as "ignore", as well
        % as those whose overlap was between negative and positive
        % overlap)
        valid_mask = regions_pos(:,6) == 1 | regions_neg(:,6) == 0;
        
        regions(~valid_mask, :) = [];
        features(:,~valid_mask) = [];
        labels = regions_pos(valid_mask, 6); % Doesn't matter if we take positive or negative output...
        
        % Add to the output
        all_features{i} = features;
        all_labels{i} = 2*labels - 1;
        
        %% Visualize training samples
        if display_svm_samples,
            fig = figure('Name', sprintf('SVM training samples: %s', basename));
            clf(fig);
            
            % Show image
            Im = uint8( bsxfun(@times, double(I), 0.50*mask + 0.50) );
            imshow(Im);
            hold on;
            
            % Draw chosen regions
            vicos.utils.draw_boxes(regions(labels == 1,:), 'color', 'green', 'line_style', '-'); % Positive
            vicos.utils.draw_boxes(regions(labels == 0,:), 'color', 'red', 'line_style', '-'); % Negative
            
            % Create fake plots for legend entries
            h = zeros(1,2);
            h(1) = plot([0,0], [0,0], '-', 'Color', 'green', 'LineWidth', 2);
            h(2) = plot([0,0], [0,0], '-', 'Color', 'red', 'LineWidth', 2);
            legend(h, 'pos', 'neg');
            
            drawnow();
        end
    end
    
    %% Train the SVM
    % Gather
    all_features = horzcat( all_features{:} );
    all_labels = vertcat( all_labels{:} );
    
    fprintf('Training SVM with %d samples, %d positive (%.2f%%), %d negative (%.2f%%)\n', numel(all_labels), sum(all_labels==1), 100*sum(all_labels==1)/numel(all_labels), sum(all_labels==-1), 100*sum(all_labels==-1)/numel(all_labels));
    
    % Train
    svm = self.classifier_factory(); % Create SVM
    svm.train(all_features, all_labels);
    
    if nargout < 1,
        self.svm_classifier = svm;
    end
end