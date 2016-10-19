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
    
    % Append positive/negative overlap settings
    identifier = [ identifier, sprintf('_pos-%g_neg-%g', self.training_positive_overlap, self.training_negative_overlap) ];
    
    % L2 normalization flag
    if self.l2_normalized_features,
        identifier = [ identifier, '_l2' ];
    end
end