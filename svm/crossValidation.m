function cv_result = crossValidation(k, features, labels, kernel_function, param1, param2, param3)
% example run: cv_result = crossValidation(k, features_c, label_c, @rbf_c, c, gamma);
% param1 = c
% param2 = gamma
% param3 = epsilon

cv_result = {};
rowsize = size(features,1);
start_index = 1;
min_loss = 1/0;
for i=1:k
    end_index = round((rowsize/k)*i);
    ftest = features(start_index:end_index,:);
    ltest = labels(start_index:end_index,1);
    ftrain = features;
    ltrain = labels;
    ftrain(start_index:end_index,:) = [];
    ltrain(start_index:end_index,:) = [];
    if exist('param3', 'var')
        mdl = kernel_function(ftrain, ltrain, param1, param2, param3);
    elseif exist('param2', 'var')
        mdl = kernel_function(ftrain, ltrain, param1, param2);
    else
        mdl = kernel_function(ftrain, ltrain, param1);
    end
    losss = loss(mdl,ftest,ltest);
    if (min_loss > losss)
        min_loss = losss;
        cv_result{1} = size(mdl.SupportVectors,1);
        cv_result{2} = min_loss;
    end
    start_index = end_index;
end
end