k = 10;

% rbf = table('Size', [0 4], 'VariableNames', {'Num Of SV Regression', 'SV % Regression', 'Num Of SV Classification', 'SV % Classification'}, 'VariableTypes', {'double', 'double', 'double', 'double'});
% Typical sigma (or gamma) and c
% 0.0001 < sigma < 10
% 0.1 < c < 100
% gamma = 1/2*(sigma(i))^2;
c = [0.1, 1, 10, 100, 1000];
gamma = [1, 0.1, 0.01, 0.001, 0.0001];
epsilon = [3,3,3,3,3];

norm_features_c = normalize(features_c);
norm_labels_c = normalize(label_c);
rbf_hp_tune_c = GridSearchCV(k, norm_features_c, norm_labels_c, @rbf_c, c, gamma);

norm_features_r = normalize(features_r);
norm_labels_r = normalize(label_r);
rbf_hp_tune_r = GridSearchCV(k, norm_features_r, norm_labels_r, @rbf_r, r, gamma, epsilon);

clear k c gamma epsilon;

% to do: fix cross validation