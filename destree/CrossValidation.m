function [k_features, k_label] = CrossValidation(features, label)
k = 10;
k_features = cell(1,k);
k_label = cell(1,k);
rowsize = size(features,1);
rand = randperm(rowsize);
shuffled_features = features(rand,:);
shuffled_label = label(rand,:);
start_index = 1;
for i=1:k
    end_index = round((rowsize/k)*i); 
    k_features{i} = shuffled_features(start_index:end_index,:);
    k_label{i} = shuffled_label(start_index:end_index);
    start_index = end_index + i;
end
end