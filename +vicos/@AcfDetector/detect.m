function [ boxes, all_boxes, time_det, time_nms ] = detect (self, I, varargin)
    % [ boxes, all_boxes, time_det, time_nms ] = DETECT (self, I, varargin)
    %
    % Runs the ACF detector on the given image.
    %
    % Input:
    %  - self: @AcfDetector instance
    %  - I: input image
    %  - varargin: key/value pairs:
    %     - nms_overlap: overlap threshold for NMS (default: 0.5)
    %
    % Output:
    %  - boxes: D1x5 array of final detections, after NMS
    %  - all_boxes: D2x5 array of all detections, before NMS
    %    (D2 >= D1)
    %  - time_det: time spent in initial detection
    %  - time_nms: time spent in the NMS
    
    parser = inputParser();
    parser.addParameter('nms_overlap', 0.5, @isnumeric);
    parser.parse(varargin{:});
    
    nms_overlap = parser.Results.nms_overlap;
    
    % Obtain all detections
    t = tic();
    all_boxes = acfDetect(I, self.detector);
    time_det = toc(t);
    
    % Separately apply NMS
    nms_params = self.nms;
    nms_params.overlap = nms_overlap;
    
    t = tic();
    boxes = bbNms(all_boxes, nms_params);
    time_nms = toc(t);
end