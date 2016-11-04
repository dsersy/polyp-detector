function set_upscale_image (self, factor)
    % SET_UPSCALE_IMAGE (self, factor)
    %
    % Sets the nOctUp parameter of detector.pyramid
    %
    % Input:
    %  - self: @AcfDetector instance
    %  - factor: new nOctUp value
    
    self.detector = acfModify(self.detector, 'nOctUp', factor);
end