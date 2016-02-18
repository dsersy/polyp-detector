function acf_train_detector (varargin)
    parser = inputParser();
    parser.addParameter('data_dir', 'acf_data', @ischar);
    parser.addParameter('name', 'Polyp', @ischar);
    parser.parse(varargin{:});
    
    data_dir = parser.Results.data_dir;
    name = parser.Results.name;
    
    %% Set up general options for training detector (see acfTrain)
    opts = acfTrain();
    opts.modelDs = [ 30, 30 ]; % Model height and width without padding
    opts.modelDsPad = [ 30, 30 ]; % Model height and width with padding
    opts.nWeak = [ 32, 128, 512, 2048 ]; % Number of weak classifiers per stage
    opts.pBoost.pTree.fracFtrs = 1/16; % parameters for boosting (see adaBoostTrain.m) 
    opts.pLoad = {};

    opts.posImgDir = fullfile(data_dir, 'train', 'pos'); % dir containing full positive images
    opts.posGtDir = fullfile(data_dir, 'train', 'posGT'); % dir containing ground truth
    opts.negImgDir = fullfile(data_dir, 'train', 'neg'); % dir containing negative images

    opts.name = name;

    %% Train
    t = tic();
    detector = acfTrain(opts);
    fprintf('Detector trained in %f seconds\n', toc(t));
end