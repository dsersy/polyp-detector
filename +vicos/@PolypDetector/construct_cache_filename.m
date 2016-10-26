function identifier = construct_cache_filename (self, basename, enhance_image, rescale_image, acf_nms_overlap)
    % identifier = CONSTRUCT_CACHE_FILENAME (self, basename, enhance_image, rescale_image, acf_nms_overlap)
    %
    % Constructs a cache basename from the given image basename and various
    % processing flags.
    
    % Image basename
    identifier = basename;
    
    % Image enhancement
    if enhance_image, 
        identifier = [ identifier, sprintf('clahe') ];
    end
    
    % Scale
    identifier = [ identifier, sprintf('-scale_%g', rescale_image) ];
    
    % ACF NMS overlap
    identifier = [ identifier, sprintf('-acf_nms_%g', acf_nms_overlap) ];
end