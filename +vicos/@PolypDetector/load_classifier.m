function load_classifier (self, filename)
    % LOAD_CLASSIFIER (self, filename)
    %
    % Loads classifier from file
    
    if ~exist('filename', 'var') || isempty(filename)
        [ filename, pathname ] = uigetfile('*.mat', 'Pick a classifier file');
        if isequal(filename, 0)
            return;
        end
        filename = fullfile(pathname, filename);
    end
    
    tmp = load(filename);
    self.svm_classifier = tmp.classifier;
end