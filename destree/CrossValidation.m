function cv_result = CrossValidation(k_tree, features_test, label_test)
% label? accuracy & evaluation score?
% return an array of prediction per fold & iteration
cell_size = size(k_tree,2);
cv_result = zeros(1,cell_size);
for i=1:cell_size
    cv_result(1,i) = PredictTree(k_tree{i}, features_test, label_test);
end
end

function accuracy = PredictTree(tree, features, label)
row_size = size(features,1);
tree_prediction = zeros(row_size,1);
accuracy_cell = zeros(row_size,1);
% apply the destree to each row, get the leaf
for i=1:row_size
    tree_prediction(i) = GetLeafValue(tree, features, i);
    accuracy_cell(i) = (label(i) - tree_prediction(i))^2;
end
% only two leafs get chosen everytime
accuracy = sum(accuracy_cell)/row_size
% use rms to get accuracy
end

function leaf_value = GetLeafValue(tree, features, i)
if ~isempty(tree.kids)
    j = tree.attribute;
    if features(i,j) < tree.threshold
        leaf_value = GetLeafValue(tree.kids{1,1}, features, i);
    elseif features(i,j) >= tree.threshold
        leaf_value = GetLeafValue(tree.kids{1,2}, features, i);
    end
else
    leaf_value = tree.prediction;
end
end