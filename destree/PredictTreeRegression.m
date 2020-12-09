function rmse = PredictTreeRegression(tree, features, label)
row_size = size(features,1);
tree_prediction = zeros(row_size,1);
accuracy_cell = zeros(row_size,1);
% apply the destree to each row, get the leaf
for i=1:row_size
    tree_prediction(i) = GetLeafValue(tree, features, i);
    accuracy_cell(i) = (label(i) - tree_prediction(i))^2;
end
% compute rmse
rmse = sqrt(mean(accuracy_cell));
end
