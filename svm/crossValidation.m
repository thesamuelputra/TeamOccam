function cv_result = crossValidation(features_train, label_train, kernel_function, param1, param2, param3)
% example run: cv_result = crossValidation(features_c, label_c, @rbf_c, c, gamma);
k = 10;
cv_result = cell(1,k);
rowsize = size(features_train,1);
rand = randperm(rowsize);
shuffled_features = features_train(rand,:);
shuffled_label = label_train(rand,:);
start_index = 1;
for i=1:k
    end_index = round((rowsize/k)*i);
    ftest = shuffled_features(start_index:end_index,:);
    ltest = shuffled_label(start_index:end_index,1);
    ftrain = features_train;
    ltrain = label_train;
    ftrain(start_index:end_index,:) = [];
    ltrain(start_index:end_index,:) = [];
    if exist('param3', 'var')
        mdl = kernel_function(shuffled_features, shuffled_label, param1(i), param2(i), param3(i))
    elseif exist('param2', 'var')  
        mdl = kernel_function(shuffled_features, shuffled_label, param1(i), param2(i))
    else
        mdl = kernel_function(shuffled_features, shuffled_label, param1(i))
    end
    accuracy = size(mdl.SupportVectors,1)/size(features_train,1);
    cv_result{i} = accuracy;
    start_index = end_index;
end
end