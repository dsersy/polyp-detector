function acf_prepare_dataset (training_images, output_path, varargin)
    parser = inputParser();
    parser.addParameter('negative_folders', {}, @iscell);
    parser.addParameter('box_scale', 1.0, @isnumeric);
    parser.addParameter('mask_images', true, @islogical);
    parser.addParameter('mix_negatives_with_positives', false, @islogical);
    parser.parse(varargin{:});

    box_scale = parser.Results.box_scale;
    negative_folders = parser.Results.negative_folders;
    mask_images = parser.Results.mask_images;
    mix_negatives_with_positives = parser.Results.mix_negatives_with_positives;
    
    %% Create output dir
    assert(exist(output_path, 'dir') == 0, 'Output directory already exists!');
    mkdir(output_path);
    mkdir(fullfile(output_path, 'train'));
    mkdir(fullfile(output_path, 'train', 'pos'));
    mkdir(fullfile(output_path, 'train', 'posGT'));
    
    %% Process positives
    fprintf('Processing positive images...\n');
    for f = 1:numel(training_images),
        [ data_path, basename, ext ] = fileparts( training_images{f} );
        
        image_file = fullfile(data_path, [ basename, ext ]);
        annotation_file = fullfile(data_path, [ basename, '.bbox' ]);
        polygon_file = fullfile(data_path, [ basename, '.poly' ]);
        
        % Parse
        fid = fopen(annotation_file, 'r');
        boxes = textscan(fid, '%f %f %f %f');
        fclose(fid);
        
        x = boxes{1};
        y = boxes{2};
        width = boxes{3};
        height = boxes{4};
        
        % Filter out invalid boxes
        invalid_idx = width == 0 | height == 0;
        x(invalid_idx) = [];
        y(invalid_idx) = [];
        width(invalid_idx) = [];
        height(invalid_idx) = [];
        
        % Enlarge box
        extra_width  = width * (box_scale - 1);
        extra_height = height * (box_scale - 1);
        
        x = x - extra_width/2;
        y = y - extra_height/2;
        width = width + extra_width;
        height = height + extra_height;
        
        num_boxes = numel(x);
        
        % Write in Dollar's format
        annotations = bbGt('create', num_boxes);
        annotations = bbGt('set', annotations, 'lbl', repmat({'polyp'}, 1, num_boxes));
        
        annotations = bbGt('set', annotations, 'bb', [ x, y, width, height ]);
        
        %% Replace . to _ in output filenames
        basename = strrep(basename, '.', '_');
        
        %% Save boxes annotation
        output_annotation = fullfile(output_path, 'train', 'posGT', [ basename, '.txt' ]);
        bbGt('bbSave', annotations, output_annotation);
        
        %% Read image, mask it, and write it to the output
        output_image = fullfile(output_path, 'train', 'pos', [ basename, '.jpg' ]);

        if mask_images,
            I = imread(image_file);
            poly = load(polygon_file);

            Im = mask_image_with_polygon(I, poly);

            imwrite(Im, output_image);
        else
            % Copy/link file
            copy_or_link_file(image_file, output_image);
        end
    end
    
    %% Process negatives
    if mix_negatives_with_positives,
        num_neg_files = 0;

        for d = 1:numel(negative_folders),
            negative_folder = negative_folders{d};
            
            fprintf('Copying negative images from "%s"...\n', negative_folder);
            files = dir(fullfile(negative_folder));
            files([files.isdir]) = [];
            for f = 1:numel(files),
                [ ~, ~, ext ] = fileparts(files(f).name);
                input_image = fullfile( negative_folder, files(f).name );
                
                basename = sprintf('negative-%04d%s', num_neg_files);
                output_image = fullfile(output_path, 'train', 'pos', [ basename, ext ]);
                output_annotation = fullfile(output_path, 'train', 'posGT', [ basename, '.txt' ]);
                        
                copy_or_link_file(input_image, output_image);
                
                annotations = bbGt('create', 0);
                bbGt('bbSave', annotations, output_annotation);
                
                num_neg_files = num_neg_files + 1;
            end
        end
    else
        negative_output_dir = fullfile(output_path, 'train', 'neg');

        if isempty(negative_folders),
            fprintf('Do not forget to copy/link the appropriate negative images to "%s"!\n', negative_output_dir);
        else
            mkdir(negative_output_dir);

            num_neg_files = 0;

            for d = 1:numel(negative_folders),
                negative_folder = negative_folders{d};
                
                fprintf('Copying negative images from "%s"...\n', negative_folder);
                files = dir(fullfile(negative_folder));
                files([files.isdir]) = [];
                for f = 1:numel(files),
                    [ ~, ~, ext ] = fileparts(files(f).name);
                    input_image = fullfile( negative_folder, files(f).name );
                    output_image = fullfile( negative_output_dir, sprintf('%04d%s', num_neg_files, ext) );
                    copy_or_link_file(input_image, output_image);
                    num_neg_files = num_neg_files + 1;
                end
            end
        end
    end
end

function Im = mask_image_with_polygon (I, poly)
    mask = poly2mask(poly(:,1), poly(:,2), size(I,1), size(I,2));
    mask = imgaussfilt(double(mask), 2);
    
    Im = uint8( bsxfun(@times, double(I), mask) );
end


