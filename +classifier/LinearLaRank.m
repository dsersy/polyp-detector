classdef LinearLaRank < handle
    % LINEARLARANK A simple forward-class to the Linear LaRank
    % implementation provided by the onyx library

    properties (Access = private)
        larank

        num_epochs
    end

    properties (SetAccess = private)
        % Map of class label strings to numeric indices
        class_labels
    end

    methods
        function self = LinearLaRank (varargin)
            % self = LINEARLARANK (varargin)
            %
            % Creates Linear LaRank SVM.
            %
            % Input:
            %  - varargin: key-value pairs, specifying the following
            %    options:
            %     - C: C parameter (default: 1.0)
            %     - tau: tau parameter (default: 0.0001)
            %     - num_epochs: number of epochs to train (default: 1)
            %
            % Output:
            %  - self: @LinearLaRank instance

            % Parse variable arguments
            parser = inputParser();
            parser.addParameter('C', 1.0, @isscalar);
            parser.addParameter('tau', 0.0001, @isscalar);
            parser.addParameter('num_epochs', 1, @isscalar);
            parser.parse(varargin{:});

            C = parser.Results.C;
            tau = parser.Results.tau;

            % Create Linear LaRank
            self.larank = onyx.LinearLaRank('C', C, 'tau', tau);

            self.num_epochs = parser.Results.num_epochs;
        end

        function identifier = get_identifier (self)
            % identifier = GET_IDENTIFIER (self)
            %
            % Returns unique classifier's unique identifier, based on
            % classifier type and its configuration.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %
            % Output:
            %  - identifier: identifier

            identifier = 'linear_larank';
        end

        function ids = get_class_ids (self)
            % ids = GET_CLASS_IDS (self)
            %
            % Returns class IDs represented by the classifier.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %
            % Output:
            %  - ids: cell array of class IDs

            % Simply gather all IDs from root nodes
            ids = self.class_labels;
        end

        function train (self, features, labels)
            % TRAIN (self, features, labels)
            %
            % Train the online classifier in batch mode.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %  - features: training features; DxN matrix, where D is
            %    feature dimension and N is number of samples
            %  - labels: 1xN cell array of training labels (strings)
            %
            % Note: this function performs batch training of the online
            % classifier, for number of epochs that was specified at
            % construction.
            %
            % Also note that this does not reset the classifier; if called
            % more than once, the subsequent calls will effectively update
            % the classifier!

            % Update labels map
            update_labels_map(self, labels);

            % Remap labels from string to numeric
            labels_numeric = labels_str2num(self, labels);

            % Batch train the classifier
            train(self.larank, features, labels_numeric, 'num_epochs', self.num_epochs);
        end

        function update (self, features, labels)
            % UPDATE (self, features, labels)
            %
            % Update the online classifier.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %  - features: training features; DxN matrix, where D is
            %    feature dimension and N is number of samples
            %  - labels: 1xN cell array of training labels (strings)

            % Update labels map
            update_labels_map(self, labels);

            % Remap labels from string to numeric
            labels_numeric = labels_str2num(self, labels);

            % Update classifier
            update(self.larank, features, labels_numeric);
        end

        function [ labels, scores, probabilities ] = predict (self, features)
            % [ label, scores, probabilities ] = PREDICT (self, features)
            %
            % Predict labels for given feature vectors.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %  - features: DxN feature vectors of samples to classify
            %
            % Output:
            %  - label: predicted label(s)
            %  - scores: CxN matrix of corresponding classification scores
            %  - probabilities: CxN matrix of classification scores
            %    converted to probabilities via an exponential function

            % Predict
            switch nargout,
                case { 0, 1 },
                    [ labels_numeric ] = predict(self.larank, features);
                case 2,
                    [ labels_numeric, scores ] = predict(self.larank, features);
                case 3,
                    [ labels_numeric, scores, probabilities ] = predict(self.larank, features);
            end

            % Remap labels from indices to strings
            labels = labels_num2str(self, labels_numeric);
        end
    end

    methods (Access = protected)
        function update_labels_map (self, labels)
            % Find new classes that are present in the input labels
            new_classes = setdiff( unique(labels), self.class_labels );

            % Append the new classes
            self.class_labels = [ self.class_labels, new_classes ];
        end

        function labels_numeric = labels_str2num (self, labels_string)
            labels_numeric = cellfun(@(x) find(ismember(self.class_labels, x)), labels_string);
            labels_numeric = labels_numeric - 1; % 0-based indexing
        end

        function labels_string = labels_num2str (self, labels_numeric)
            labels_string = self.class_labels(labels_numeric + 1); % 0-based index
        end
    end
end

