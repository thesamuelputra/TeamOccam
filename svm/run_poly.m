k = 10;

c = [0.1, 1, 10, 100, 1000];
q = [0.1, 1, 10, 100, 1000];
epsilon = [3,3,3,3,3];

norm_features_c = normalize(features_c);
norm_labels_c = normalize(labels_c);
poly_hp_tune_c = GridSearchCV(k, norm_features_c, norm_labels_c, @poly_c, c, q);

norm_features_r = normalize(features_r);
norm_labels_r = normalize(labels_r);
poly_hp_tune_r = GridSearchCV(k, norm_features_r, norm_labels_r, @poly_r, c, q, epsilon);

clear k q epsilon;

% different parameters for regre and clasi?