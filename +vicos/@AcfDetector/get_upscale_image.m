function factor = get_upscale_image (self)
    % factor = GET_UPSCALE_IMAGE (self, factor)
    %
    % Retrieves the values of nOctUp parameter of detector.pyramid
    %
    % Input:
    %  - self: @AcfDetector instance
    %
    % Output:
    %  - factor: nOctUp value
    
    factor = self.detector.opts.pPyramid.nOctUp;
end