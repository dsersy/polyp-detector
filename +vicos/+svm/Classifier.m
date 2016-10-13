classdef Classifier < handle
    % CLASSIFIER General binary SVM classifier interface
    %
    % (C) 2016 Rok Mandeljc
    
    methods (Abstract)
        % identifier = GET_IDENTIFIER (self)
        %
        % Returns unique classifier's unique identifier, based on
        % classifier type and its configuration.
        %
        % Input:
        %  - self: @Classifier instance
        %
        % Output:
        %  - identifier: identifier
        identifier = get_identifier (self)
            
        % TRAIN (self, features, labels)
        %
        % Train the classifier in batch mode.
        %
        % Input:
        %  - self: @Classifier instance
        %  - features: training features; DxN matrix, where D is
        %    feature dimension and N is number of samples
        %  - labels: 1xN vector of training labels (-1/+1)
        train (self, features, labels)
        
        % [ label, scores ] = PREDICT (self, features)
        %
        % Predict labels for given feature vectors.
        %
        % Input:
        %  - self: @Classifier instance
        %  - features: DxN feature vectors of samples to classify
        %
        % Output:
        %  - labels: 1xN cell array of predicted label(s)
        %  - scores: CxN matrix of corresponding classification scores
        [ labels, scores ] = predict (self, features)
    end
end