function [best_attribute, best_threshold, attributes] = ChooseAttributeRegression(features, label)
% measures how “good” each attribute (i.e. feature) in the set is.
% implements RSS from CART algorithm
attributes = zeros(2,size(features,2));
for i = 1:size(features,2)
    if ~(any(isnan(features(:,i))))
        [attributes(1,i), attributes(2,i)] = getThreshold(features(:,i), label);
    else
        attributes(1,i) = NaN;
        attributes(2,i) = NaN;
    end
end
best_attribute = find(attributes(1,:)==min(attributes(1,:)),1,'first'); % return the column index
best_threshold = attributes(2, best_attribute);
end

function [min_rss,threshold] = getThreshold(col, label)
min_rss = 1/0;
rss = 0;
threshold = col(1);
for i = 1:(length(label) - 1)
    % split into two group by i and find the rss of each row
    r1 = label(1:i);
    r2 = label((i+1):length(label));
    for j = 1:i
        rss = rss + (label(j) - mean(r1))^2;
    end
    for k = (i+1):length(col)
        rss = rss + (label(k) - mean(r2))^2;
    end
    if (rss < min_rss)
        min_rss = rss;
        % choose the row with minimum rss as the threshold
        threshold = col(i);
    end
    rss = 0;
end
end