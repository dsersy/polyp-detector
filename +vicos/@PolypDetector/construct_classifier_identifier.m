function identifier = construct_classifier_identifier (self)
    % filename = CONSTRUCT_CLASSIFIER_IDENTIFIER (self)
    %
    % Constructs classifier identifier, based on classifier type and 
    % relevant settings.
    %
    % Input:
    %  - self:
    %
    % Output:
    %  - identifier: a string that hopefully uniquely identifiers the
    %    classifier type and settings
    
    % Create temporary instance via classifier factory
    classifier = self.classifier_factory();
    identifier = classifier.get_identifier();
    
    % Enhanced image or not
    if self.enhance_image,
        identifier = [ identifier, sprintf('-clahe') ];
    end
    
    % Append positive/negative overlap settings
    identifier = [ identifier, sprintf('-acf_nms_%g-pos_%g-neg_%g', self.acf_nms_overlap, self.training_positive_overlap, self.training_negative_overlap) ];
    
    % L2 normalization flag
    if self.l2_normalized_features,
        identifier = [ identifier, '-norm_l2' ];
    end
end