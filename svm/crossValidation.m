function cv_result = crossValidation(features_train, label_train, classification, c, gamma)
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
    if classification
        % Typical sigma (or gamma) and c
        % 0.0001 < sigma < 10
        % 0.1 < c < 100
        % gamma = 1/2*(sigma(i))^2;
        cv_result{i} = rbf_c(shuffled_features, shuffled_label, c(i), gamma(i));
        
    end
    start_index = end_index;
end
end