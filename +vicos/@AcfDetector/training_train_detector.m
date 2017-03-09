function detector = training_train_detector (data_dir, varargin)
    % TRAINING_TRAIN_DETECTOR (data_dir, varargin)
    %
    % Trains an ACF detector.
    %
    % Input:
    %  - data_dir: training dataset directory, prepared using
    %    TRAINING_PREPARE_DATASET()
    %  - varargin: optional key/value pairs
    %     - name: detector name (default: 'Polyp')
    %     - window_size: window size (default: [ 32, 32 ])
    %     - padded_size: padded window size (default: []; use window_size)
    %     - output_file: store detector to the specified file
    %
    % Output:
    %  - detector: trained detector structure

    % Input parsing
    parser = inputParser();
    parser.addParameter('name', 'Polyp', @ischar);
    parser.addParameter('window_size', [ 30, 30 ], @isnumeric);
    parser.addParameter('padded_size', [], @isnumeric);
    parser.addParameter('output_file', '', @ischar);
    parser.parse(varargin{:});

    name = parser.Results.name;
    window_size = parser.Results.window_size;
    padded_size = parser.Results.padded_size;
    if isempty(padded_size)
        padded_size = window_size;
    end
    output_file = parser.Results.output_file;

    assert(exist(data_dir, 'dir') ~= 0, 'Invalid dataset directory!');

    %% Set up general options for training detector (see acfTrain)
    opts = acfTrain();
    opts.modelDs = window_size; % Model height and width without padding
    opts.modelDsPad = padded_size; % Model height and width with padding
    opts.nWeak = [ 32, 128, 512, 2048 ]; % Number of weak classifiers per stage
    opts.pBoost.pTree.fracFtrs = 1/16; % parameters for boosting (see adaBoostTrain.m)
    opts.pLoad = {};

    % Split boxes in batches of 5000 to improve NMS performance
    % (otherwise training tends to take forever)
    opts.pNms.maxn = 5000;

    % Specify dataset
    opts.posImgDir = fullfile(data_dir, 'train', 'pos'); % dir containing full positive images
    opts.posGtDir = fullfile(data_dir, 'train', 'posGT'); % dir containing ground truth

    negative_dir = fullfile(data_dir, 'train', 'neg');
    if exist(negative_dir, 'dir')
        opts.negImgDir = negative_dir; % dir containing negative images
    end

    % Store detector and log to temporary directory
    opts.name = tempname();

    %% Train
    t = tic();
    detector = acfTrain(opts);
    training_time = toc(t);
    fprintf('Detector trained in %f seconds\n', training_time);

    detector.opts.name = name; % Override name

    % Save detector?
    if ~isempty(output_file)
        save(output_file, '-v7.3', 'detector', 'training_time');
    end
end
