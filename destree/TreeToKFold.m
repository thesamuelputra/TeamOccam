function [k_tree, cv_result] = TreeToKFold(features_train, label_train, headers, max_depth)
k = 10;
cv_result = zeros(1,k);
k_tree = cell(1,k);
rowsize = size(features_train,1);
rand = randperm(rowsize);
shuffled_features = features_train(rand,:);
shuffled_label = label_train(rand,:);
start_index = 1;
for i=1:k
    end_index = round((rowsize/k)*i);
    ftest = shuffled_features(start_index:end_index,:);
    ltest = shuffled_label(start_index:end_index);
    ftrain = features_train;
    ltrain = label_train;
    ftrain(start_index:end_index,:) = [];
    ltrain(start_index:end_index) = [];
    k_tree{i} = DecisionTreeLearning(ftrain, ltrain, headers, max_depth);
    cv_result(1,i) = PredictTree(k_tree{i}, ftest, ltest);
    start_index = end_index;
end
end