function make ()
    root_dir = fileparts(mfilename('fullpath'));
    
    %% liblinear
    liblinear_dir = fullfile(root_dir, 'external', 'liblinear', 'matlab');
    fprintf('\nCompiling liblinear ...');
    run(fullfile(liblinear_dir, 'make.m'));

    target_dir = fullfile(root_dir, '+classifier', '@LIBLINEAR', 'private');
    fprintf('Copying liblinear MEX files to "%s"...\n', target_dir);
    copy_liblinear_files(liblinear_dir, target_dir);
end

function copy_liblinear_files (source_dir, target_dir)
    liblinear_files_to_copy = { [ 'train.', mexext() ],   [ 'liblinear_train.', mexext() ];
                                [ 'predict.', mexext() ], [ 'liblinear_predict.', mexext() ] };
    for f = 1:size(liblinear_files_to_copy, 1),
        copyfile(fullfile(source_dir, liblinear_files_to_copy{f, 1}), fullfile(target_dir, liblinear_files_to_copy{f, 2}), 'f');
    end
end