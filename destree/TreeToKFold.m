function k_tree = TreeToKFold(features_train, label, headers, max_depth)
k = 10;
k_tree = cell(1,k);
rowsize = size(features_train,1);
rand = randperm(rowsize);
shuffled_features = features_train(rand,:);
shuffled_label = label(rand,:);
start_index = 1;
for i=1:k
    end_index = round((rowsize/k)*i);
    f = shuffled_features(start_index:end_index,:);
    l = shuffled_label(start_index:end_index);
    k_tree{i} = DecisionTreeLearning(f, l, headers, max_depth);
    start_index = end_index + i;
end
end