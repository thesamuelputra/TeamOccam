function threshold =  ChooseThreshold(features,targets)
% measures how “good” each attribute (i.e. feature) in the set is.
% choose best attribute (from all of the column) and best threshold (?)
threshold = zeros(1,2);
% size(features,2)
for i = 1:size(features,2)
    fprintf('Calculating column...');
    disp(i);
    threshold(i) = getThreshold(features(:,i));
end
disp('threshold = ');
disp(threshold);

% test for first column
% col = features(:,2);
% disp(getThreshold(col));
% CART?
% GINI IMPURITY
end

function threshold = getThreshold(col)
min_rss = max(col);
rss = 0;
threshold = 0;
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
    if (rss > min_rss)
        min_rss = rss;
        % choose the row with minimum rss as the threshold
        threshold = col(i);
    end
    rss = 0;
end
end