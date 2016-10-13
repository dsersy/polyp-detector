function boxes = enlarge_boxes (boxes, scale_factor)
    % boxes = ENLARGE_BOXES (boxes, scale_factor)
    
    % Load
    x = boxes(:,1);
    y = boxes(:,2);
    w = boxes(:,3);
    h = boxes(:,4);
    
    % Modify
    extra_width  = w * (scale_factor - 1);
    extra_height = h * (scale_factor - 1);
        
    x = x - extra_width/2;
    y = y - extra_height/2;
    w = w + extra_width;
    h = h + extra_height;
    
    % Store
    boxes(:,1) = x;
    boxes(:,2) = y;
    boxes(:,3) = w;
    boxes(:,4) = h;
end