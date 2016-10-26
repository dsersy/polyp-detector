function Ie = adaptive_histogram_equalization (I, varargin)
    % Ie = ADAPTIVE_HISTOGRAM_EQUALIZATION (I, varargin)
    %
    % Enhances image by performing adaptive histogram equalization on its
    % intensity channel. Internally, it converts an RGB image to HSV, calls
    % adapthisteq on the third channel, and converts the image back to RGB.
    %
    % Input:
    %  - I: input image (grayscale or RGB)
    %  - varargin: key/value arguments to be passed to ADAPTIVEHISTEQ
    %
    % Output:
    %  - Ie: enhanced image of same dimensions and type as the input image
    
    if size(I, 3) == 1,
        % Grayscale
        Ie = adapthisteq(I, varargin{:});
    else
        % RGB -> HSV
        Ihsv = rgb2hsv(I);

        % CLAHE on intensity
        Ihsv(:,:,3) = adapthisteq(Ihsv(:,:,3), varargin{:});

        % HSV -> RGB
        Ie = hsv2rgb(Ihsv);

        % Convert back to uint8
        if isa(I, 'uint8'),
            Ie = uint8(255*Ie);
        end
    end
end