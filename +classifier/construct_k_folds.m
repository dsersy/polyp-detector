function folds = construct_k_folds (labels, K)
    % folds = CONSTRUCT_K_FOLDS (labels, K)
    %
    % Constructs K folds for cross-validation, while preserving the ratio
    % of classes within the folds.
    %
    % Input:
    %  - labels: 1xN vector of labels
    %  - K: number of folds
    %
    % Output:
    %  - folds: 1xK cell array, where each cell contains a vector of
    %    indices corresponding to the specified fold
    %
    % (C) 2014, Rok Mandeljc <rok.mandeljc@fe.uni-lj.si>

    % Determine number of classes
    classes = unique(labels);
    
    % Shuffle indices
    indices = cell(1, numel(classes));
    for c = 1:numel(classes),
        indices{c} = find(labels == classes(c));
        indices{c} = indices{c}(randperm(numel(indices{c})));
    end
    
    % Construct folds into temporary cell array
    tmp = cell(numel(classes), K);
    for k = 1:K,
        for c = 1:numel(classes),
            tmp{c,k} = indices{c}(k:K:numel(indices{c}));
        end
    end
    
    % Merge indices for each fold
    folds = cell(1, K);
    for k = 1:K,
        folds{k} = [ tmp{:,k} ];
    end
end