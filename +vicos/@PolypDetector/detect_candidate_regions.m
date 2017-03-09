function [ regions, time_det, time_nms ] = detect_candidate_regions (self, I, cache_file)
    % [ regions, time_det, time_nms ] = DETECT_CANDIDATE_REGIONS (self, I, cache_file)
    %
    % Detects candidate regions in the given image.
    %
    % Input:
    %  - self:
    %  - I: image
    %  - cache_file: optional cache file to use (default: '')
    %
    % Output:
    %  - regions: detected regions
    %  - time_det: time spent in region detection
    %  - time_nms: time spent in the first non-maxima suppression
    %    pass
    %
    % Note: creates ACF instance on demand
    
    if ~exist('cache_file', 'var')
        cache_file = '';
    end
    
    %% Region detection / caching
    if ~isempty(cache_file) && exist(cache_file, 'file')
        % Load from cache
        tmp = load(cache_file);
        
        % Validate cache file
        assert(self.acf_nms_overlap == tmp.nms_overlap, 'Invalid cache file; non-maxima suppression overlap threshold mismatch!');
        
        % Copy from cache
        regions = tmp.regions;
        time_det = tmp.time_det;
        time_nms = tmp.time_nms;
    else
        % Create ACF detector, if necessary
        if isempty(self.acf_detector)
            self.acf_detector = self.acf_detector_factory();
        end
        
        % Run ACF detector
        [ regions, regions_all, time_det, time_nms ] = self.acf_detector.detect(I, 'nms_overlap', self.acf_nms_overlap);
        
        % Save to cache
        if ~isempty(cache_file)
            vicos.utils.ensure_path_exists(cache_file);
            
            tmp = struct(...
                'nms_overlap', self.acf_nms_overlap, ...
                'regions', regions, ...
                'regions_all', regions_all, ...
                'time_det', time_det, ...
                'time_nms', time_nms); %#ok<NASGU>
            
            save(cache_file, '-v7.3', '-struct', 'tmp');
        end
    end
    
    % Rescale boxes, if necessary
    if self.acf_box_scaling ~= 1.0
        regions = enlarge_boxes(regions, self.acf_box_scaling);
    end
end