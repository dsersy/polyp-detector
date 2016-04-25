root_dir = fileparts(mfilename('fullpath'));

% This folder
addpath(root_dir);

% Piotr's toolbox
addpath(genpath( fullfile(root_dir, 'external', 'piotr_toolbox') ));

% CNN feature extractor
run( fullfile(root_dir, 'external', 'cnn-feature-extractor', 'startup.m' ) );
