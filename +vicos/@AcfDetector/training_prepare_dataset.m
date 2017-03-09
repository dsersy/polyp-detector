function training_prepare_dataset (training_images, output_path, varargin)
    % TRAINING_PREPARE_DATASET (training_images, output_path, varargin)
    %
    % Prepares dataset for training an ACF detector.
    %
    % Input:
    %  - training_images: cell array of input training image names
    %  - output_path: output directory with prepared dataset
    %  - varargin: optional key/value pairs
    %     - negative_folders: cell array with paths to folders that contain
    %       negative images
    %     - box_scale: rescaling of annotated bounding boxes (default: 1.0)
    %     - mask_images: boolean flag indicating whether the images should
    %       be masked with ROI polygon or not (default: true)
    %     - ignore_masked_boxes: boolean flag indicating whether the boxes
    %       that intersect with masked areas should be marked as ignored or
    %       not (default: true)
    %     - mix_negatives_with_positives: a logical flag indicating whether
    %       negative images should be mixed with the positive ones
    %       (default: false). If set to true, the negative images are
    %       placed into train/pos folder, and empty annotation files are
    %       generated for them. Consequently, the detector training
    %       procedure will perform hard negative mining on positive images
    %       as well. On the other hand, if set to false, the negative
    %       images are placed into train/neg folder, and negatives will be
    %       mined only from those images.
    %     - force_copy: always copy images instead of creating symbolic
    %       links on supported platforms (default: false)
    %
    % Note: for images that are not masked (the negative images, or if
    % masking is turned off), symbolic links are created on linux systems
    % for performance reasons. On Windows, such images are copied.
    
    % Input parser
    parser = inputParser();
    parser.addParameter('negative_folders', {}, @iscell);
    parser.addParameter('box_scale', 1.0, @isnumeric);
    parser.addParameter('mask_images', true, @islogical);
    parser.addParameter('ignore_masked_boxes', true, @islogical);
    parser.addParameter('mix_negatives_with_positives', false, @islogical);
    parser.addParameter('force_copy', false, @islogical);
    parser.addParameter('enhance_images', false, @islogical);
    parser.parse(varargin{:});

    box_scale = parser.Results.box_scale;
    negative_folders = parser.Results.negative_folders;
    mask_images = parser.Results.mask_images;
    ignore_masked_boxes = parser.Results.ignore_masked_boxes;
    mix_negatives_with_positives = parser.Results.mix_negatives_with_positives;
    force_copy = parser.Results.force_copy;
    enhance_images = parser.Results.enhance_images;
    
    %% Create output dir
    assert(exist(output_path, 'dir') == 0, 'Output directory already exists!');
    mkdir(output_path);
    mkdir(fullfile(output_path, 'train'));
    mkdir(fullfile(output_path, 'train', 'pos'));
    mkdir(fullfile(output_path, 'train', 'posGT'));
    
    %% Process positives
    for f = 1:numel(training_images)
        image_file = training_images{f};
        
        %% Load data
        % We make use of static methods from PolypDetector class for
        % consistency
        [ I, ~, poly, boxes ] = vicos.PolypDetector.load_data(image_file);
        [ ~, basename, ext ] = fileparts(image_file);

        % Remove invalid boxes (the ones with with or height equal to 0)
        invalid_idx = boxes(:,3) == 0 | boxes(:,4) == 0;
        boxes(invalid_idx,:) = [];
        
        % Rescale boxes (no-op if box_scale is 1.0)
        boxes = vicos.PolypDetector.enlarge_boxes(boxes, box_scale);

        % Mark boxes on the edge of ROI as ignore
        if ignore_masked_boxes
            validity_mask = poly2mask(poly(:,1), poly(:,2), size(I, 1), size(I,2));
            
            x1 = boxes(:,1);
            y1 = boxes(:,2);
            x2 = boxes(:,1) + boxes(:,3) + 1;
            y2 = boxes(:,2) + boxes(:,4) + 1;
        
            x1 = max(min(round(x1), size(validity_mask, 2)), 1);
            y1 = max(min(round(y1), size(validity_mask, 1)), 1);
            x2 = max(min(round(x2), size(validity_mask, 2)), 1);
            y2 = max(min(round(y2), size(validity_mask, 1)), 1);
        
            valid_tl = validity_mask( sub2ind(size(validity_mask), y1, x1) );
            valid_tr = validity_mask( sub2ind(size(validity_mask), y1, x2) );
            valid_bl = validity_mask( sub2ind(size(validity_mask), y2, x1) );
            valid_br = validity_mask( sub2ind(size(validity_mask), y2, x2) );
        
            ignore_flags = ~(valid_tl & valid_tr & valid_bl & valid_br);
        else
            ignore_flags = zeros(size(boxes, 1), 1);
        end
        
        % Write in Dollar's format
        num_boxes = size(boxes, 1);
        
        annotations = bbGt('create', num_boxes);
        annotations = bbGt('set', annotations, 'lbl', repmat({'polyp'}, 1, num_boxes));
        annotations = bbGt('set', annotations, 'bb', boxes);
        annotations = bbGt('set', annotations, 'ign', ignore_flags);
        
        %% Replace . to _ in output filenames
        % It seems that ACF detector training takes issue with dots in the
        % filenames...
        basename = strrep(basename, '.', '_');
        
        %% Save boxes annotation
        output_annotation = fullfile(output_path, 'train', 'posGT', [ basename, '.txt' ]);
        bbGt('bbSave', annotations, output_annotation);
        
        %% Save/copy image
        if mask_images || enhance_images
            Io = I;
            
            % Enhance?
            if enhance_images
                Io = vicos.utils.adaptive_histogram_equalization(Io);
            end
            
            % Mask?
            if mask_images
                Io = vicos.PolypDetector.mask_image_with_polygon(Io, poly);
            end
            
            % Save the modified copy
            output_image = fullfile(output_path, 'train', 'pos', [ basename, '.jpg' ]);
            imwrite(Io, output_image);
        else
            % Copy/link file
            output_image = fullfile(output_path, 'train', 'pos', [ basename, ext ]);
            copy_or_link_file(image_file, output_image, force_copy);
        end
    end
    
    %% Process negatives
    if mix_negatives_with_positives
        num_neg_files = 0;

        for d = 1:numel(negative_folders)
            negative_folder = negative_folders{d};
            
            fprintf('Copying negative images from "%s"...\n', negative_folder);
            files = dir(fullfile(negative_folder));
            files([files.isdir]) = [];
            for f = 1:numel(files)
                [ ~, ~, ext ] = fileparts(files(f).name);
                input_image = fullfile( negative_folder, files(f).name );
                
                basename = sprintf('negative-%04d%s', num_neg_files);
                output_image = fullfile(output_path, 'train', 'pos', [ basename, ext ]);
                output_annotation = fullfile(output_path, 'train', 'posGT', [ basename, '.txt' ]);

                if enhance_images
                    Ii = imread(input_image);
                    Io = vicos.utils.adaptive_histogram_equalization(Ii);
                    imwrite(Io, output_image);
                else
                    copy_or_link_file(input_image, output_image, force_copy);
                end
                
                annotations = bbGt('create', 0);
                bbGt('bbSave', annotations, output_annotation);
                
                num_neg_files = num_neg_files + 1;
            end
        end
    else
        negative_output_dir = fullfile(output_path, 'train', 'neg');

        if isempty(negative_folders)
            fprintf('Do not forget to copy/link the appropriate negative images to "%s"!\n', negative_output_dir);
        else
            mkdir(negative_output_dir);

            num_neg_files = 0;

            for d = 1:numel(negative_folders)
                negative_folder = negative_folders{d};
                
                fprintf('Copying negative images from "%s"...\n', negative_folder);
                files = dir(fullfile(negative_folder));
                files([files.isdir]) = [];
                for f = 1:numel(files)
                    [ ~, ~, ext ] = fileparts(files(f).name);
                    input_image = fullfile( negative_folder, files(f).name );
                    output_image = fullfile( negative_output_dir, sprintf('%04d%s', num_neg_files, ext) );
                    
                    if enhance_images
                        Ii = imread(input_image);
                        Io = vicos.utils.adaptive_histogram_equalization(Ii);
                        imwrite(Io, output_image);
                    else
                        copy_or_link_file(input_image, output_image, force_copy);
                    end
                    
                    num_neg_files = num_neg_files + 1;
                end
            end
        end
    end
end

function copy_or_link_file (input_filename, output_filename, force_copy)
    % COPY_OR_LINK_FILE (input_filename, output_filename, force_copy)
    %
    % Copies/links the input file to output file.
    %
    % Input:
    %  - input_filename: input filename
    %  - output_filename: output filename
    %  - force_copy: copy the file, even if symbolic links are supported on
    %    the platform
    %
    % Note: symbolic links are available only on linux. On other platforms,
    % a copy is always made.
    
    if ~force_copy && isunix()
        command = sprintf('ln -s "$(readlink -f "%s")" "%s"', input_filename, output_filename);
        system(command);
    else
        copyfile(input_filename, output_filename);
    end
end