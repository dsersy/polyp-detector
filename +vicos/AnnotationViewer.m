classdef AnnotationViewer < handle
    % ANNOTATIONVIEWER - Annotation viewer utility
    %
    % A utility to view and edit polyp annotations.
    %
    % (C) 2016 Rok Mandeljc <rok.mandeljc@fri.uni-lj.si>
    
    properties
        % Filenames
        image_file
        boxes_file
        poly_file
        
        % Loaded data
        I
        boxes
        polygon
        
        % Handles
        figure_main
        
        impoly_handle % Handle for impoly polygon drawing
        
        polygon_handles
        boxes_handles
        
        % Box scaling
        boxes_scale = 1
        
        % Box selection        
        selected_boxes
        selected_box_idx
        selected_box_handle
        
        % Image masking switch
        show_masked = false
    end
    
    methods
        function self = AnnotationViewer ()
            % self = ANNOTATIONVIEWER ()
            %
            % Creates an instance of annotation viewer.
            
            % Create main figure
            self.figure_main = figure('Interruptible', 'off', 'BusyAction', 'cancel');
            set(self.figure_main, 'WindowKeyPressFcn', @(fig_obj, event_data) self.window_key_press(event_data));
            set(self.figure_main, 'WindowButtonDownFcn', @(fig_obj, event_data) self.window_mouse_button(event_data));
            
            % Turn off the image size warning
            warning('off', 'Images:initSize:adjustingMag');
            
            fprintf('Annotation viewer ready; press "h" for help.\n');
        end
        
        function load_image (self, filename)
            % LOAD_IMAGE (self, filename)
            %
            % Load an image. If filename is not provided, file selection
            % dialog is shown.
            %
            % Input:
            %  - self:
            %  - filename: image file to load
            
            % Select file
            if ~exist('filename', 'var') || isempty(filename)
                % Select file
                [ filename, pathname ] = uigetfile('*.jpg;*.png;*.bmp;*.jpeg;*.tif', 'Pick an image file', self.image_file);
                if isequal(filename, 0)
                    return;
                end
                filename = fullfile(pathname, filename);
            end
            
            % Store image filename, and construct annotations and polygon
            % filename. These are used just for suggesting the output
            % filenames; for actual loading, we use the
            % vicos.PolypDetector.load_data() method.
            
            self.image_file = filename;
            [ pathname, basename, ~ ] = fileparts(self.image_file);

            self.boxes_file = fullfile(pathname, [ basename, '.bbox' ]);
            self.poly_file = fullfile(pathname, [ basename, '.poly' ]);

            % Load data, without filtering the loaded bounding boxes
            [ self.I, ~, self.polygon, self.boxes ] = vicos.PolypDetector.load_data(self.image_file);
            
            % Display image name
            set(self.figure_main, 'Name', sprintf('%s', basename));
            
            % Reset bounding box selection            
            self.selected_box_idx = [];
            self.selected_boxes = [];
            
            % Refresh data
            self.display_data();
        end
        
        function load_next_image (self, direction)
            % LOAD_NEXT_IMAGE (self, direction)
            %
            % Loads the previous/next image located in the same folder as 
            % the current image. 
            %
            % Input:
            %  - self:
            %  - direction: direction to look for the image (-1: previous,
            %    +1 next)
            
            % Decompose image file to get path and extension
            [ path, basename, ext ] = fileparts(self.image_file);
            
            % List all images of same type in the path
            files = dir(fullfile(path, [ '*', ext ]));
            files = { files.name };
            
            % Find current image
            idx = find(strcmp([ basename, ext ], files));
            
            % Select previous/next
            new_idx = idx + direction;
            
            if new_idx < 1 || new_idx > numel(files)
                return;
            end
            
            % Load the image
            self.load_image(fullfile(path, files{new_idx}));
        end
        
        function display_data (self)
            % DISPLAY_DATA (self)
            %
            % Display all data.
            
            % Clear
            figure(self.figure_main);
            clf(self.figure_main);
            
            % Display image
            if self.show_masked
                imshow(vicos.PolypDetector.mask_image_with_polygon(self.I, self.polygon));
            else
                imshow(self.I);
            end
            hold on;
            
            %% Display polygon
            self.display_polygon();
    
            %% Display boxes
            self.display_boxes();
        end
        
        function display_boxes (self)
            % DISPLAY_BOXES (self)
            %
            % Display bounding box annotations.
            
            % Remove old boxes
            idx = ishandle(self.boxes_handles);
            delete(self.boxes_handles(idx));
            
            % Display new boxes
            self.boxes_handles = vicos.utils.draw_boxes(...
                vicos.PolypDetector.enlarge_boxes(self.boxes, self.boxes_scale), ...
                'Color', 'red');
            
            % Reshape handles to allow individual selection
            self.boxes_handles = reshape(self.boxes_handles, [], 4);
        end
        
        function display_selected_box (self)
            % DISPLAY_SELECTED_BOX (self)
            %
            % Display currently-selected bounding box.
            
            if ~isempty(self.selected_box_handle)
                valid_idx = ishandle(self.selected_box_handle);
                delete(self.selected_box_handle(valid_idx));
            end
            
            if ~isempty(self.selected_boxes)
                idx = self.selected_boxes(self.selected_box_idx);
                
                % Select box
                selected_box = vicos.PolypDetector.enlarge_boxes(self.boxes(idx,:), self.boxes_scale);
                
                self.selected_box_handle = vicos.utils.draw_boxes(selected_box, 'color', 'red', 'line_width', 2); 
            end
        end
        
        function display_polygon (self)
            % DISPLAY_POLYGON (self)
            %
            % Display ROI polygon.
            
            if ~isempty(self.polygon_handles) && ishandle(self.polygon_handles),
                delete(self.polygon_handles);
                self.polygon_handles = [];
            end

            % Do we even have a polygon?
            polygon = self.polygon;
            if isempty(polygon)
                return;
            end
            
            % Make sure polygon is closed
            if ~isequal(polygon(1,:), polygon(end,:))
                polygon = [ polygon; polygon(1,:) ]; % Make sure polygon is closed
            end
            
            % Draw new polygon
            self.polygon_handles = plot(polygon(:,1), polygon(:,2), 'y-', 'LineWidth', 2);
        end
        
        function modify_polygon (self)
            % MODIFY_POLYGON (self)
            %
            % Modify the ROI polygon.
            
            figure(self.figure_main);
            self.impoly_handle = impoly();
        end
        
        function select_next_box (self, offset)
            % SELECT_NEXT_BOX (self, offset)
            %
            % Selects previous/next bounding box annotation.
            
            % Do we have a box selected?
            if isempty(self.selected_boxes) || isempty(self.selected_box_idx)
                return;
            end
            
            % Select next box
            self.selected_box_idx = self.selected_box_idx + offset;
            
            % Wrap-around
            if self.selected_box_idx > numel(self.selected_boxes)
                self.selected_box_idx = 1;
            end
            if self.selected_box_idx < 1
                self.selected_box_idx = numel(self.selected_boxes);
            end
            
            % Update bounding box selection
            self.display_selected_box();
        end
        
        function delete_selected_box (self)
            % DELETE_SELECTED_BOX (self)
            %
            % Deletes currently selected bounding box annotation.
            
            % Do we have a box selected?
            if isempty(self.selected_boxes)
                return;
            end
            
            idx = self.selected_boxes(self.selected_box_idx);
                        
            % Clear selection
            delete(self.selected_box_handle);
            self.selected_box_handle = [];
                        
            self.selected_boxes = [];
            self.selected_box_idx = [];
                        
            % Delete the box
            self.boxes(idx,:) = [];
                        
            delete(self.boxes_handles(idx,:));
            self.boxes_handles(idx,:) = [];
        end
        
        function add_box (self)
            % ADD_BOX (self)
            %
            % Add a bounding box annotation.
            
            % Allow user to draw a rectangle
            figure(self.figure_main);
            rect = imrect();
            
            % If drawn rectangle is valid, undo the currently-set scaling,
            % and store it.
            if isvalid(rect)
                % Get box (x, y, w, h)
                box = rect.getPosition();
                
                % Undo scaling
                w = box(3) / self.boxes_scale;
                h = box(4) / self.boxes_scale;
                
                x = box(1) + (box(3) - w)/2;
                y = box(2) + (box(4) - h)/2;
                
                % Append
                self.boxes(end+1,:) = [ x, y, w, h ];
                
                %% Draw
                % We re-apply the scaling, just to be sure...
                added_box = vicos.PolypDetector.enlarge_boxes(self.boxes(end,:), self.boxes_scale);
                self.boxes_handles(end+1,:) = vicos.utils.draw_boxes(added_box, 'color', 'red');
                
                delete(rect);
            end
        end
        
        function filter_annotations (self)
            % FILTER_ANNOTATIONS (self)
            %
            % Filters out invalid annotations.
            
            % Filter out boxes with invalid dimensions
            area = self.boxes(:,3) .* self.boxes(:,4);
            invalid_mask = area == 0;
            fprintf('Removing boxes with invalid dimensions (%d)\n', sum(invalid_mask));
            self.boxes(invalid_mask, :) = [];
            
            % Filter out exact duplicates
            unique_boxes = unique(self.boxes, 'rows', 'stable');
            fprintf('Filtered out diplicates: %d -> %d\n', size(self.boxes, 1), size(unique_boxes, 1));
            self.boxes = unique_boxes;
        end
        
        function display_help (self)
            % DISPLAY_HELP (self)
            %
            % Displays help message.
            
            fprintf('\nAnnotation viewer - keyboard shortcuts:\n');
            fprintf(' l: select an image to load\n');
            fprintf(' a, ->: load next image in folder\n');
            fprintf(' s, <-: load previous image in folder\n');
            fprintf(' e: set box scaling factor\n');
            fprintf(' p: modify ROI polygon\n');
            fprintf(' -, delete: remove selected bounding box\n');
            fprintf(' +: add a bounding box\n');
            fprintf(' m: switch image masking\n');
            fprintf(' 1: export bounding box annotations\n');
            fprintf(' 2: export ROI polygon\n');
            fprintf(' f: filter bounding boxes\n');
            fprintf(' h: display this message\n');
        end
        
        function window_key_press (self, event)
            % WINDOW_KEY_PRESS (self, event)
            %
            % Keyboard event handler.
            
            switch event.Key
                case { 'a', 'leftarrow' }
                    self.load_next_image(-1);
                case { 's', 'rightarrow' }
                    self.load_next_image(+1);
                case 'l'
                    self.load_image();
                case 'e'
                    prompt = {'Enter box scale factor:'};
                    name = 'Scale boxes';
                    numlines = 1;
                    defaultanswer = {num2str(self.boxes_scale)};
 
                    answer = inputdlg(prompt, name, numlines, defaultanswer);
                    
                    if isempty(answer)
                        return;
                    end
                    
                    self.boxes_scale = str2double(answer{1});
                    
                    self.display_boxes();
                case 'p'
                    self.modify_polygon()
                case 'return'
                    if ~isempty(self.impoly_handle) && isvalid(self.impoly_handle)
                        self.polygon = self.impoly_handle.getPosition();
                        delete(self.impoly_handle);
                        self.impoly_handle = [];
                        
                        self.display_polygon();
                    end     
                case { 'delete', 'subtract' }
                    % Delete box
                    self.delete_selected_box();
                case { 'add' }
                    self.add_box();
                case 'm'
                    self.show_masked = ~self.show_masked;
                    self.display_data();
                case '1'
                    % 1: save annotations
                    [ filename, pathname ] = uiputfile(self.boxes_file, 'Save boxes as');
                    if isequal(filename, 0)
                        return;
                    end
                    filename = fullfile(pathname, filename);
                    
                    fid = fopen(filename, 'w');
                    fprintf(fid, '%g %g %g %g\n', self.boxes');
                    fclose(fid);
                case '2'
                    % 2: save polygon
                    [ filename, pathname ] = uiputfile(self.poly_file, 'Save polygon as');
                    if isequal(filename, 0)
                        return;
                    end
                    filename = fullfile(pathname, filename);
                    
                    fid = fopen(filename, 'w');
                    fprintf(fid, '%g %g\n', self.polygon');
                    fclose(fid);
                case 'f'
                    % Filter annotations
                    self.filter_annotations();
                    self.display_data();
                case 'h'
                    self.display_help();
            end
        end
        
        function window_mouse_button (self, event_data)
            % WINDOW_MOUSE_BUTTON (self, event_data)
            %
            % Mouse event handler.
            
            % Get cursor position
            figure(self.figure_main);
            pos = get(gca(), 'CurrentPoint');
            x = pos(1,1);
            y = pos(1,2);
                
            % Find all intersecting detections
            all_boxes = vicos.PolypDetector.enlarge_boxes(self.boxes, self.boxes_scale);
            x1 = all_boxes(:,1);
            y1 = all_boxes(:,2);
            x2 = all_boxes(:,1) + all_boxes(:,3);
            y2 = all_boxes(:,2) + all_boxes(:,4);
                            
            idx = find(x >= x1 & x <= x2 & y >= y1 & y <= y2);
             
            if isequal(idx, self.selected_boxes)
                select_next_box(self, +1);
            else
                % Store list of all boxes
                self.selected_boxes = idx;
    
                % Reset the selection
                self.selected_box_idx = 1;
            end

            self.display_selected_box();
        end
    end
end