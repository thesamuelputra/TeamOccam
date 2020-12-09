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