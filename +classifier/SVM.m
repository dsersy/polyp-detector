classdef SVM < handle
    % SVM An abstract SVM classifier class
    %
    % This is an abstract SVM classifier class. It provides functionality
    % such as K-fold cross-validation and Platt probabilistic calibration
    % for SVM implementations that do not have this functionality built in.
    %
    % (C) 2014 Rok Mandeljc <rok.mandeljc@fe.uni-lj.si>

    properties
        % Model
        model

        % Platt's scaling parameters
        A
        B

        % SVM regularization parameter setting
        C
    end

    methods (Access = public)
        function accuracy = train (self, features, labels)
            % accuracy = TRAIN (self, features, labels)
            %
            % Trains a SVM classifier.
            %
            % Input:
            %  - self: @SVM instance
            %  - features: training features; DxN matrix, where D is
            %    feature dimension and N is number of samples
            %  - labels: training labels; 1xN vector (+1/-1)
            %
            % Output:
            %  - prediction accuracy on 3-fold cross-validation

            assert(isrow(labels) || iscolumn(labels), 'labels must be 1xN vector!');
            assert(size(features, 2) == numel(labels), 'features must be DxN matrix!');

            % Transpose labels, if necessary
            if iscolumn(labels),
                labels = labels';
            end
            
            predicted_scores = [];
            if isempty(self.C),
                % Find optimal C parameter using 3-fold cross-validation;
                % we can reuse the predicted scores for probabilistic
                % calibration as well!
                [ self.C, predicted_scores ] = find_optimal_c_parameter(self, features, labels, 3);
            end

            % SVM training using SVM implementation
            self.model = svm_train_impl(self, features, labels);

            % If we do not have cross-validation predicted scores already
            % (from search for optimal C), obtain them now, as we need them
            % for Platt probabilistic calibration and accuracy computation
            if isempty(predicted_scores),
                predicted_scores = k_fold_cross_validation(self, features, labels, 3);
            end

            % Probabilistic calibration
            [ self.A, self.B ] = classifier.platt_calibration(predicted_scores, labels);

            % Cross-validation accuracy
            accuracy = sum((predicted_scores >= 0) * 2 - 1 == labels) / numel(labels);
        end

        function [ labels, scores, probabilities ] = predict (self, features)
            assert(~isempty(self.model), 'SVM is not trained!');

            % Predict scores using SVM implementation
            scores = svm_predict_impl(self, self.model, features);

            % Convert scores to labels
            labels = (scores >= 0) * 2 - 1;

            % Compute probabilities
            if nargout >= 3,
                assert(~isempty(self.A) && ~isempty(self.B), 'SVM is not probabilistically calibrated!');
                probabilities = 1 ./ (1 + exp(self.A * scores + self.B));
            end
        end
    end
    
    % Abstract methods implemented by children
    methods (Abstract, Access = public)
        identifier = get_identifier (self)
    end
    
    methods (Abstract, Access = protected)
        model = svm_train_impl (self, features, labels)
        scores = svm_predict_impl (self, model, features)
    end

    methods (Access = protected)
        function [ decision_scores ] = k_fold_cross_validation (self, features, labels, K)
            % Allocate decision scores
            decision_scores = nan(1, numel(labels));

            % Create folds
            folds = classifier.construct_k_folds(labels, K);
            
            % K-fold cross validation
            for k = 1:K,
                % Gather indices
                test_fold = k;
                train_folds = setdiff(1:K, k);
                
                test_idx = [ folds{test_fold} ];
                train_idx = [ folds{train_folds} ];

                % Train
                tmp_model = svm_train_impl(self, features(:, train_idx), labels(train_idx));

                % Test
                decision_scores(test_idx) = svm_predict_impl(self, tmp_model, features(:,test_idx));
            end
        end

        function [ opt_C, opt_scores ] = find_optimal_c_parameter (self, features, labels, K)
            C_values = 10 .^ (-2:2); % C values in grid search

            max_accuracy = 0;
            opt_C = [];
            opt_scores = [];

            % FIXME: reseed RNG to always use same folds?

            for c = 1:numel(C_values),
                % Temporarily set C
                self.C = C_values(c);

                % Perform K-fold cross validation
                predicted_scores = k_fold_cross_validation (self, features, labels, K);

                accuracy = sum((predicted_scores >= 0) * 2 - 1 == labels) / numel(labels);

                if accuracy > max_accuracy,
                    max_accuracy = accuracy;
                    opt_C = C_values(c);
                    opt_scores = predicted_scores;
                end
            end
        end
    end
end
