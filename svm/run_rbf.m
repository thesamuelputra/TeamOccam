k = 10;

% rbf = table('Size', [0 4], 'VariableNames', {'Num Of SV Regression', 'SV % Regression', 'Num Of SV Classification', 'SV % Classification'}, 'VariableTypes', {'double', 'double', 'double', 'double'});
% Typical sigma (or gamma) and c
% 0.0001 < sigma < 10
% 0.1 < c < 100
% gamma = 1/2*(sigma(i))^2;
% c = [0.1, 1, 10, 100, 1000, 0.1, 1, 10, 100, 1000];
% gamma = [1, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.001, 0.01, 0.1, 1];
c = [0.1, 1, 10, 100, 1000];
gamma = [1, 0.1, 0.01, 0.001, 0.0001];

epsilon = [3,3,3,3,3];

rbf_hpt_c = gridSearchCV(k, features_c, labels_c, @rbf_c, c, gamma);

norm_features_r = normalize(features_r);
norm_labels_r = normalize(labels_r);
rbf_hpt_r = gridSearchCV(k, norm_features_r, norm_labels_r, @rbf_r, c, gamma, epsilon);

rbf_hpt_c{6,7} = 'CLASSRATE';
rbf_hpt_r{6,7} = 'RMSE';
svm_rbf_classrate = zeros(1,5);
svm_rbf_rmse = zeros(1,5);
for i=1:5
    svm_rbf_classrate(i) = getClassRate(rbf_hpt_c{i,6}, labels_c);
    svm_rbf_rmse(i) = getRMSE(rbf_hpt_r{i,6}, norm_labels_r);
    rbf_hpt_c{i,7} = getClassRate(rbf_hpt_c{i,6}, labels_c);
    rbf_hpt_r{i,7} = getRMSE(rbf_hpt_r{i,6}, norm_labels_r);
end

% model = rbf_c(norm_features_c, norm_labels_c, 0.1, 1);
% disp(model);

clear k c gamma epsilon;

% to do: fix cross validation