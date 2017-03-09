function [ features, time ] = extract_features_from_regions (self, I, regions, cache_file)
    % features = EXTRACT_FEATURES_FROM_REGIONS (self, I, regions)
    %
    % Extract CNN features from given regions.
    %
    % Input:
    %  - self:
    %  - I: image
    %  - regions: Nx4 matrix describing regions
    %  - cache_file: optional cache file to use (default: '')
    %
    % Output:
    %  - features: DxN matrix of extracted features
    %  - time: time spent in feature extraction
    %
    % Note: creates CNN Feature Extractor instance on demand
    
    if ~exist('cache_file', 'var')
        cache_file = '';
    end
    
    %% Feature extraction / caching
    if ~isempty(cache_file) && exist(cache_file, 'file')
        % Load from cache
        tmp = load(cache_file);
        
        % Validate cache file
        assert(size(regions, 1) == size(tmp.features, 2), 'Invalid cache file; mismatch between number of regions and stored feature vectors!');
        
        % Copy from cache
        features = tmp.features;
        time = tmp.time;
    else
        % Convert [ x, y, w, h ] to [ x1, y1, x2, y2 ], in 4xN format
        boxes = [ regions(:,1), regions(:,2), regions(:,1)+regions(:,3)+1, regions(:,2)+regions(:,4)+1 ]';
        
        % Create extractor, if necessary
        if isempty(self.cnn_extractor)
            self.cnn_extractor = self.feature_extractor_factory();
        end
        
        % Extract CNN features
        [ features, time ] = self.cnn_extractor.extract(I, 'regions', boxes);
        
        % Save to cache
        if ~isempty(cache_file)
            vicos.utils.ensure_path_exists(cache_file);
            
            tmp = struct(...
                'features', features, ...
                'time', time); %#ok<NASGU>
            
            save(cache_file, '-v7.3', '-struct', 'tmp');
        end
    end
    
    %% Feature normalization
    if self.l2_normalized_features
        features = bsxfun(@rdivide, features, sqrt(sum(features .^ 2)));
    end
end