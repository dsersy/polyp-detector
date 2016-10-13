function startup ()
    % Root directory
    root_dir = fileparts(mfilename('fullpath'));

    %% This folder
    addpath(root_dir);

    %% Piotr's toolbox
    addpath(genpath( fullfile(root_dir, 'external', 'piotr_toolbox') ));

    %% LIBLINEAR
    addpath( fullfile(root_dir, 'external', 'liblinear', 'matlab') );

    %% lapjv
    addpath( fullfile(root_dir, 'external', 'lapjv') );
    
    %% CNN feature extractor
    run( fullfile(root_dir, 'external', 'cnn-feature-extractor', 'startup.m' ) );
end
