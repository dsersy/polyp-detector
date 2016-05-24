classdef AnnotationViewer < handle
    properties
        image_file
        boxes_file
        poly_file
        
        I
        boxes
        polygon
        
        polygon_handles
        boxes_handles
        
        boxes_scale = 1
        
        figure_main
        poly_handle
        
        selected_boxes
        selected_box_idx
        selected_box_handle
        
        show_masked = false
    end
    
    methods
        function self = AnnotationViewer ()
            self.figure_main = figure();
            set(self.figure_main, 'WindowKeyPressFcn', @(fig_obj, event_data) window_key_press(self, event_data));
            set(self.figure_main, 'WindowButtonDownFcn', @(fig_obj, event_data) window_mouse_button(self, event_data));

        end
        
        function load_image (self)
            % Select file
            [ filename, pathname ] = uigetfile('*.jpg;*.png;*.bmp;*.jpeg;*.tif', 'Pick am image file', self.image_file);
            if isequal(filename, 0),
                return;
            end
            
            self.image_file = fullfile(pathname, filename);
            [ pathname, basename, ~ ] = fileparts(self.image_file);

            % Load image
            set(self.figure_main, 'Name', sprintf('%s', basename));
            self.I = imread(self.image_file);
            
            % Load annotations
            self.boxes_file = fullfile(pathname, [ basename, '.bbox' ]);
            if exist(self.boxes_file, 'file'),
                self.boxes = load(self.boxes_file);
            else
                self.boxes = [];
            end
            
            self.selected_box_idx = [];
            self.selected_boxes = [];
            
            % Load polygon
            self.poly_file = fullfile(pathname, [ basename, '.poly' ]);
            if exist(self.poly_file, 'file'),
                self.polygon = load(self.poly_file);
            else
                self.polygon = [];
            end
                        
            % Refresh data
            self.display_data();
        end
        
        function [ x1, y1, x2, y2 ] = get_boxes (self)
            if isempty(self.boxes),
                x1 = [];
                y1 = [];
                x2 = [];
                y2 = [];
                return;
            end
            
            % Enlarge box
            extra_width  = self.boxes(:,3)' * (self.boxes_scale - 1);
            extra_height = self.boxes(:,4)' * (self.boxes_scale - 1);
    
            % Annotations
            x1 = self.boxes(:,1)' - extra_width/2;
            x2 = x1 + self.boxes(:,3)' + extra_width;
            y1 = self.boxes(:,2)' - extra_height/2;
            y2 = y1 + self.boxes(:,4)' + extra_height;
        end
        
        function display_data (self)
            % Clear
            figure(self.figure_main);
            clf(self.figure_main);
            
            % Display image
            if self.show_masked,
                imshow(mask_image_with_polygon(self.I, self.polygon));
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
            % Remove old boxes
            idx = ishandle(self.boxes_handles);
            delete(self.boxes_handles(idx));
            
            % Display new boxes
            [ x1, y1, x2, y2 ] = self.get_boxes();
            
            self.boxes_handles = line([ x1, x1, x1, x2;
                                        x2, x2, x1, x2 ], ...
                                      [ y1, y2, y1, y1;
                                        y1, y2, y2, y2 ], 'Color', 'red');
            self.boxes_handles = reshape(self.boxes_handles, [], 4);
        end
        
        function display_selected_box (self)
            if ~isempty(self.selected_box_handle),
                valid_idx = ishandle(self.selected_box_handle);
                delete(self.selected_box_handle(valid_idx));
            end
            
            if ~isempty(self.selected_boxes),
                idx = self.selected_boxes(self.selected_box_idx);
                
                [ x1, y1, x2, y2 ] = self.get_boxes();
                x1 = x1(idx);
                y1 = y1(idx);
                x2 = x2(idx);
                y2 = y2(idx);
                
                self.selected_box_handle = line([ x1, x1, x1, x2;
                                                  x2, x2, x1, x2 ], ...
                                                [ y1, y2, y1, y1;
                                                  y1, y2, y2, y2 ], ...
                                                  'Color', 'red', 'LineWidth', 2); 
            end
        end
        
        function display_polygon (self)
            if ~isempty(self.polygon_handles) && ishandle(self.polygon_handles),
                delete(self.polygon_handles);
                self.polygon_handles = [];
            end
            
            % Draw new polygon
            polygon = self.polygon;
            if ~isequal(polygon(1,:), polygon(end,:)),
                polygon = [ polygon; polygon(1,:) ]; % Make sure polygon is closed
            end
            self.polygon_handles = plot(polygon(:,1), polygon(:,2), 'y-', 'LineWidth', 2);
        end
        
        function modify_polygon (self)
            figure(self.figure_main);
            self.poly_handle = impoly();
        end
        
        function select_next_box (self, offset)
            if isempty(self.selected_boxes) || isempty(self.selected_box_idx),
                return;
            end
            
            % Select next box
            self.selected_box_idx = self.selected_box_idx + offset;
            
            % Wrap-around
            if self.selected_box_idx > numel(self.selected_boxes),
                self.selected_box_idx = 1;
            end
            if self.selected_box_idx < 1,
                self.selected_box_idx = numel(self.selected_boxes);
            end
            
            % Update bounding box selection
            self.display_selected_box();
        end
        
        function delete_selected_box (self)
            if ~isempty(self.selected_boxes),
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
        end
        
        function add_box (self)
            figure(self.figure_main);
            rect = imrect();
            if isvalid(rect),
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
                % This is sub-optimal, but on the other hand ensures that
                % all scaling has been properly accounted for
                [ x1, y1, x2, y2 ] = self.get_boxes();
                x1 = x1(end); y1 = y1(end); x2 = x2(end); y2 = y2(end);
                
                self.boxes_handles(end+1,:) = line([ x1, x1, x1, x2;
                                                     x2, x2, x1, x2 ], ...
                                                   [ y1, y2, y1, y1;
                                                     y1, y2, y2, y2 ], 'Color', 'red');
                
                delete(rect);
            end
        end
        
        function filter_annotations (self)
            % FILTER_ANNOTATIONS (self) 
            
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
        
        function window_key_press (self, event)
            switch event.Key,
                case 'l',
                    self.load_image();
                case 'e',
                    prompt = {'Enter box scale factor:'};
                    name = 'Scale boxes';
                    numlines = 1;
                    defaultanswer = {num2str(self.boxes_scale)};
 
                    answer = inputdlg(prompt, name, numlines, defaultanswer);
                    
                    if isempty(answer),
                        return;
                    end
                    
                    self.boxes_scale = str2double(answer{1});
                    
                    self.display_boxes();
                case 'p',
                    self.modify_polygon()
                case 'return',
                    if ~isempty(self.poly_handle) && isvalid(self.poly_handle),
                        self.polygon = self.poly_handle.getPosition();
                        delete(self.poly_handle);
                        self.poly_handle = [];
                        
                        self.display_polygon();
                    end     
                case { 'delete', 'subtract' },
                    % Delete box
                    self.delete_selected_box();
                case { 'a', 'add' },
                    self.add_box();
                case 'm',
                    self.show_masked = ~self.show_masked;
                    self.display_data();
                case '1',
                    % 1: save annotations
                    [ filename, pathname ] = uiputfile(self.boxes_file, 'Save boxes as');
                    if isequal(filename, 0),
                        return;
                    end
                    filename = fullfile(pathname, filename);
                    
                    fid = fopen(filename, 'w');
                    fprintf(fid, '%g %g %g %g\n', self.boxes');
                    fclose(fid);
                case '2',
                    % 2: save polygon
                    [ filename, pathname ] = uiputfile(self.poly_file, 'Save polygon as');
                    if isequal(filename, 0),
                        return;
                    end
                    filename = fullfile(pathname, filename);
                    
                    fid = fopen(filename, 'w');
                    fprintf(fid, '%g %g\n', self.polygon');
                    fclose(fid);
                case 'f',
                    % Filter annotations
                    self.filter_annotations();
                    self.display_data();
            end
        end
        
        function window_mouse_button (self, event_data)
            % Get cursor position
            figure(self.figure_main);
            pos = get(gca(), 'CurrentPoint');
            x = pos(1,1);
            y = pos(1,2);
                
            % Find all intersecting detections
            [ x1, y1, x2, y2 ] = self.get_boxes();
                
            idx = find(x >= x1 & x <= x2 & y >= y1 & y <= y2);
             
            if isequal(idx, self.selected_boxes),
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