function Im = mask_image_with_polygon (I, poly)
    % Im = MASK_IMAGE_WITH_POLYGON (I, poly)
    %
    % Masks the input image with polygon.
    %
    % Input:
    %  - I: input image (HxWx3, uint8)
    %  - poly: ROI polygon (Nx2 matrix)
    %
    % Output:
    %  - Im: output image (HxWx3, uint8)
    
    mask = poly2mask(poly(:,1), poly(:,2), size(I,1), size(I,2));
    mask = imgaussfilt(double(mask), 2);
    Im = uint8( bsxfun(@times, double(I), mask) );
end