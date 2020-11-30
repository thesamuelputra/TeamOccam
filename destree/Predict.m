function result = Predict(tree, features)
row_size = size(features,1);
result = zeros(row_size,1);
% apply the destree to each row, get the leaf
for i=1:row_size
    fprintf('Calculating row...');
    disp(i);
    result(i) = GetPrediction(tree, features, i);
end
% only two leafs get chosen everytime
end

function prediction = GetPrediction(tree, features, i)
if ~isempty(tree.kids)
    j = tree.attribute;
    if features(i,j) < tree.threshold
        prediction = GetPrediction(tree.kids{1,1}, features, i);
    elseif features(i,j) >= tree.threshold
        prediction = GetPrediction(tree.kids{1,2}, features, i);
    end
else
    prediction = tree.prediction;
end
end