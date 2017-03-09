function load_acf_detector (self, filename)
    % LOAD_ACF_DETECTOR (self, filename)
    %
    % Loads an ACF detector from file
    
    if ~exist('filename', 'var') || isempty(filename)
        [ filename, pathname ] = uigetfile('*.mat', 'Pick an ACF detector file');
        if isequal(filename, 0)
            return;
        end
        filename = fullfile(pathname, filename);
    end
    
    self.acf_detector = vicos.AcfDetector(filename);
end