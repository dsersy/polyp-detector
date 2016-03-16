classdef LIBLINEAR < classifier.SVM
    % LIBLINEAR A wrapper class for linear SVM provided by LIBLINEAR package
    %
    % (C) 2014 Rok Mandeljc <rok.mandeljc@fe.uni-lj.si>

    properties
        % LIBLINEAR bias
        bias
    end
    
    methods (Access = public)
        function self = LIBLINEAR (varargin)
            % self = LIBLINEAR (varargin)
            %
            % Creates LIBLINEAR-based linear SVM.
            %
            % Input:
            %  - varargin: key-value pairs, specifying the following
            %    options:
            %     - C: C parameter. If left empty (default), the optimal
            %       value is found during training using cross-validation.
            %
            % Output:
            %  - self: @LIBLINEAR instance

            % Make sure we have pre-requisites
            assert(exist('liblinear_train', 'file') == 3, 'liblinear_train() does not seem to exist! Compile LIBLINEAR and move its train and predict MEX files to private/liblinear_train and private/liblinear_predict!');

            % Call superclass constructor
            self = self@classifier.SVM();

            % Parse variable arguments
            parser = inputParser();
            parser.addParameter('C', [], @isscalar);
            parser.addParameter('bias', [], @isscalar);
            parser.parse(varargin{:});

            self.C = parser.Results.C;
            self.bias = parser.Results.bias;
        end
    end
    
    % Methods required by classifier.SVM
    methods (Access = public)
        function identifier = get_identifier (self)
            identifier = 'liblinear';
        end
    end

    methods (Access = protected)
        function model = svm_train_impl (self, features, labels)
            liblinear_opts = '-q ';
            if ~isempty(self.C),
                liblinear_opts = [ liblinear_opts, sprintf('-c %f ', self.C) ];
            end
            if ~isempty(self.bias),
                liblinear_opts = [ liblinear_opts, sprintf('-B %f ', self.bias) ];
            end
            model = liblinear_train(labels', sparse(double(features)), liblinear_opts, 'col');
        end

        function scores = svm_predict_impl (self, model, features)
            if model.bias >= 0,
                scores = model.w(1:end-1) * features + model.w(end)*model.bias;
            else
                scores = model.w * features;
            end
        end
    end
end
