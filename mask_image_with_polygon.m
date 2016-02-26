function Im = mask_image_with_polygon (I, poly)
    mask = poly2mask(poly(:,1), poly(:,2), size(I,1), size(I,2));
    mask = imgaussfilt(double(mask), 2);
    
    Im = uint8( double(I).*repmat(mask, 1, 1, size(I,3)) );
end

