function handles = draw_boxes (boxes, varargin)
    % handles = DRAW_BOXES (boxes, varargin)
    %
    % Draws multiple colored boxes.
    %
    % Input:
    %  - boxes: Nx4 matrix ([ x, y, w, h ])
    %  - varargin: optional key/value pairs
    %     - color: box color (default: red)
    %     - line_style: line style (default: -)
    %     - line_width: line width (default: 1.0)
    %
    % Output:
    %  - handles: graphic handles
   
    % Input arguments
    parser = inputParser();
    parser.addParameter('color', 'red', @ischar);
    parser.addParameter('line_style', '-', @ischar);
    parser.addParameter('line_width', 1.0, @isnumeric);
    parser.parse(varargin{:});
    
    color = parser.Results.color;
    line_style = parser.Results.line_style;    
    line_width = parser.Results.line_width;    

    % [ x, y, w, h ] -> [ x1, y1, x2, y2 ]
    x1 = boxes(:,1)';
    y1 = boxes(:,2)';
    x2 = boxes(:,1)' + boxes(:,3)' + 1;
    y2 = boxes(:,2)' + boxes(:,4)' + 1;
            
    % Draw boxes
    handles = line([ x1, x1, x1, x2;
                     x2, x2, x1, x2 ], ...
                   [ y1, y2, y1, y1;
                     y1, y2, y2, y2 ], 'Color', color, 'LineStyle', line_style, 'LineWidth', line_width);
end
