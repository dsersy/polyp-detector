function ensure_path_exists (file)
    % ENSURE_PATH_EXISTS (file)
    %
    % Ensures that the parent directory of the specified file exists.
    %
    % Input:
    %  - file: filename
    
    path = fileparts(file);
    if ~exist(path, 'dir'),
        mkdir(path);
    end
end