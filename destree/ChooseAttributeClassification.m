function [best_attribute, best_threshold] = ChooseAttributeClassification(features)
% implements Gini impurity from CART algorithm
attributes = zeros(2,size(features,2));
for i = 1:size(features,2)
    [attributes(1,i), attributes(2,i)] = getThreshold(features(:,i), label);
end
best_attribute = find(attributes(1,:)==min(attributes(1,:)),1,'first'); % return the column index
best_threshold = attributes(2, best_attribute);
end

function [min_gini, threshold] = getThreshold(col, array_y)
% Merge then sort then split
table = [col, array_y];
table = sortrows(table, 1);
col = table(:,1);
array_y = table(:,2);

min_gini = 1/0;
% gini = 1 - probno^2 - probyes^2;
threshold = col(1);

for i = 1:(length(col)-1)
%   for calc gini in continuous data 
    avg_weight = (col(i) + col(i+1)) / 2;
    prob_l_pos = sum(array_y(1:i) == 1);
    prob_l_neg = sum(array_y(1:i) == -1);
    prob_l_total = length(array_y(1:i));
    gini_l = 1 - (prob_l_pos/(prob_l_total))^2 - (prob_l_neg/(prob_l_total))^2;
    prob_r_pos = sum(array_y((i+1):length(col)) == 1);
    prob_r_neg = sum(array_y((i+1):length(col)) == -1);
    prob_r_total = length(array_y((i+1):length(col)));
    gini_r = 1 - (prob_r_pos/(prob_r_total))^2 - (prob_r_neg/(prob_r_total))^2;
    total_gini = (prob_l_total/(prob_l_total+prob_r_total))*gini_l + (prob_r_total/(prob_l_total+prob_r_total))*gini_r;
    if (total_gini < min_gini)
        min_gini = total_gini;
        % choose the row with minimum gini as the threshold
        threshold = avg_weight;
    end
end
end
