classdef LibLinear < vicos.svm.Classifier
    % LIBLINEAR Binary linear SVM
    %
    % Wrapper for binary linear SVM from LIBLINEAR package.
    %
    % (C) 2016, Rok Mandeljc <rok.mandeljc@fe.uni-lj.si>

    properties
        C
        bias
        
        % Weight samples inversely to their frequency (in order to balance
        % them out)
        weight_samples
        
        % SVM
        model
    end

    methods (Access = public)
        function self = LibLinear (varargin)
            % self = LIBLINEAR (varargin)
            %
            % Creates a binary linear SVM.
            %
            % Input:
            %  - varargin: key-value pairs, specifying the following
            %    options:
            %     - C: regularization parameter; if empty (default), the
            %       optimal value is estimated using LIBLINEAR built-in
            %       search
            %     - bias: bias; if empty (default), the LIBLINEAR-default
            %       value is used
            %     - weight_samples: whether to weight the samples inversely
            %       to the frequency of their class (to balance them).
            %       Default: false
            %
            % Output:
            %  - self:

            % Make sure 'train' and 'predict' are MEX files in the path (if
            % not, LIBLINEAR is likely not available)
            assert(exist('train', 'file') == 3, '"train" from LIBLINEAR not found in the path!');
            assert(exist('predict', 'file') == 3, '"predict" from LIBLINEAR not found in the path!');

            % Parse variable arguments
            parser = inputParser();
            parser.addParameter('C', [], @isscalar);
            parser.addParameter('bias', [], @isscalar);
            parser.addParameter('weight_samples', false, @isscalar);
            parser.parse(varargin{:});

            self.C = parser.Results.C;
            self.bias = parser.Results.bias;
            self.weight_samples = parser.Results.weight_samples;
        end
    end
    
    % svm.Classifier
    methods (Access = public)
        function identifier = get_identifier (self)
            identifier = 'liblinear';
            if self.weight_samples
                identifier = [ identifier, '_weighted' ];
            end
        end
        
        function train (self, features, labels)
            assert(isrow(labels) || iscolumn(labels), 'labels must be a 1xN or Nx1 vector!');
            assert(isnumeric(labels), 'labels must be a numeric vector!');
            assert(all(ismember(labels, [ 1, -1 ])), 'labels must be -1/+1!');
            assert(size(features, 2) == numel(labels), 'features must be DxN matrix!');
            
            if iscolumn(labels)
                labels = labels';
            end
            
            % LIBLINEAR options; we use primal formulation of L2-regularized
            % L2-loss SVC because it supports automatic estimation of the 
            % C parameter...
            liblinear_opts = '-q -s 2';
            
            % Sample weighting
            if self.weight_samples
                num_pos = sum(labels ==  1);
                num_neg = sum(labels == -1);
                
                w_pos = (num_pos + num_neg) / (2 * num_pos);
                w_neg = (num_pos + num_neg) / (2 * num_neg);
                                
                liblinear_opts = [ liblinear_opts, sprintf(' -w-1 %g -w1 %g', w_neg, w_pos) ];
            end
            
            % Bias
            if ~isempty(self.bias)
                liblinear_opts = [ liblinear_opts, sprintf(' -B %f ', self.bias) ];
            end
            
            % C; if not provided, estimate via cross-validation
            if isempty(self.C)
                best = train(labels', sparse(double(features)), [ liblinear_opts, ' -C' ], 'col');
                optC = best(1);
            else
                optC = self.C;
            end
            
            liblinear_opts = [ liblinear_opts, sprintf(' -c %f ', optC) ];
            
            % Train the final model
            self.model = train(labels', sparse(double(features)), liblinear_opts, 'col');
        end

        function [ labels, scores ] = predict (self, features)
            num_samples = size(features, 2);
            
            [ labels, ~, scores ] = predict(-1*ones(num_samples, 1), sparse(double(features)), self.model, '-q', 'col');
            
            labels = labels'; % 1xN
            scores = scores'; % 1xN
        end
    end
end
