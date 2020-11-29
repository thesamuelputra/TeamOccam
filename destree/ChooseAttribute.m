% something is wrong with the function, attributes return the same col over
% and over again.

function [best_attribute, best_threshold, attributes] = ChooseAttribute(features)
% measures how “good” each attribute (i.e. feature) in the set is.
attributes = zeros(2,size(features,2));
for i = 1:size(features,2)
    [attributes(1,i), attributes(2,i)] = getThreshold(features(:,i));
end
best_attribute = find(attributes(1,:)==min(attributes(1,:)),1,'first'); % return the column index
best_threshold = attributes(2, best_attribute);
fprintf('Best Attribute...');
disp(best_attribute);
fprintf('Best Threshold...');
disp(best_threshold);
end

function [min_rss,threshold] = getThreshold(col)
min_rss = 1/0;
rss = 0;
threshold = col(1);
rss_t_array = zeros(2, length(col) - 1);
for i = 1:(length(col) - 1)
    % split into two group by i and find the rss of each row
    r1 = col(1:i);
    r2 = col((i+1):length(col));
    for j = 1:i
        rss = rss + (col(j) - mean(r1))^2;
    end
    for k = (i+1):length(col)
        rss = rss + (col(k) - mean(r2))^2;
    end
    if (rss < min_rss)
        min_rss = rss;
        % choose the row with minimum rss as the threshold
        threshold = col(i);
    end
    rss_t_array(1, i) = rss;
    rss_t_array(2, i) = col(i);
    rss = 0;
end
% disp(rss_t_array)
% disp(threshold)
% disp(min_rss)
% plot(rss_t_array(2,:), rss_t_array(1,:), 'ro');
end