function cv_result = CrossValidation(k_tree, features_train, features_test)
% label? accuracy & evaluation score?
% return an array of prediction per fold & iteration
cell_size = size(k_tree,2);
cv_result = zeros(cell_size,cell_size);
for i=1:cell_size
    for j=1:cell_size
        if j == i
            features = features_test;
        else
            features = features_train;
        end
        cv_result(j,i) = PredictTree(k_tree{j}, features);
    end
end
end

function tree_prediction = PredictTree(tree, features)
row_size = size(features,1);
tree_prediction = zeros(row_size,1);
% apply the destree to each row, get the leaf
for i=1:row_size
    tree_prediction(i) = GetLeafValue(tree, features, i);
end
% only two leafs get chosen everytime
tree_prediction = mean(tree_prediction);
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