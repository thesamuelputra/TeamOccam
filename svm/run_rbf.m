k = 10

rbf = table('Size', [0 2], 'VariableNames', {'Regression', 'Classification'}, 'VariableTypes', {'double', 'double'});
% Typical sigma (or gamma) and c
% 0.0001 < sigma < 10
% 0.1 < c < 100
% gamma = 1/2*(sigma(i))^2;
c = [1,2,3,4,5,6,7,8,9,10];
gamma = [1,2,3,4,5,6,7,8,9,10];
epsilon = [3,3,3,3,3,3,3,3,3,3];
cvr_r = crossValidation(features_r, label_r, @rbf_r, c, gamma, epsilon);
cvr_c = crossValidation(features_c, label_c, @rbf_c, c, gamma);

for i=1:k
    disp(i);
    rbf = [rbf; {cvr_r{i}, cvr_c{i}}];
    disp(rbf);
end

clear i k c gamma epsilon;