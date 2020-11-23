function [best_attribute,best_threshold] = ChooseAttribute(features,label)
% measures how “good” each attribute (i.e. feature) in the set is.
% choose best attribute (from all of the column) and best threshold (?)
% choose attribute = lowest rss become root?
% best_attribute and best_threshold come from the same column?
threshold = zeros(1,2);
min_rss = zeros(1,2);
for i = 1:size(features,2)
    fprintf('Calculating threshold column...');
    disp(i);
    [min_rss(i), threshold(i)] = getThreshold(features(:,i));
end
best_attribute = min(min_rss);
best_threshold = min(threshold);
disp(find(min(min_rss)));
disp(find(min(threshold)));
end

function [min_rss,threshold] = getThreshold(col)
min_rss = max(col);
rss = 1/0;
threshold = 1/0;
for i = 1:length(col)
    % split into two group by i and find the rss of each row
    r1 = col(1:i);
    r2 = col(i:length(col));
    for j = 1:length(r1)
        rss = rss + (col(j) - mean(r1))^2;
    end
    for k = 1:length(r2)
        rss = rss + (col(j) - mean(r2))^2;
    end
    if (rss < min_rss)
        min_rss = rss;
        % choose the row with minimum rss as the threshold
        threshold = col(i);
    end
    rss = 0;
end
end