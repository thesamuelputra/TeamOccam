k = 10;

c = [0.1, 1, 10, 100, 1000];
q = [0.1, 1, 10, 100, 1000];
epsilon = [3,3,3,3,3];

poly_hpt_c = gridSearchCV(k, features_c, labels_c, @poly_c, c, q);

norm_features_r = normalize(features_r);
norm_labels_r = normalize(labels_r);
poly_hpt_r = gridSearchCV(k, norm_features_r, norm_labels_r, @poly_r, c, q, epsilon);

poly_hpt_c{6,7} = 'CLASSRATE';
poly_hpt_r{6,7} = 'RMSE';
svm_poly_classrate = zeros(1,5);
svm_poly_rmse = zeros(1,5);
for i=1:5
    svm_poly_classrate(i) = getClassRate(poly_hpt_c{i,6}, labels_c);
    svm_poly_rmse(i) = getRMSE(poly_hpt_r{i,6}, norm_labels_r);
    poly_hpt_c{i,7} = getClassRate(poly_hpt_c{i,6}, labels_c);
    poly_hpt_r{i,7} = getRMSE(poly_hpt_r{i,6}, norm_labels_r);
end

clear k q epsilon;

% different parameters for regre and clasi?