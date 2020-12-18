function hp_tune_result = GridSearchCV(k_fold, features, labels, kernel_function, param1, param2, param3)
% GridSearchCV for hyperparameter tuning

length_param = length(param1);
hp_tune_result = zeros(length_param, 5); % c, gamma/poly, epsilon, cv_result
for i=1:length_param
    p1 = param1(i);
    hp_tune_result(i,1) = p1;
    if exist('param3', 'var')
        p2 = param2(i);
        p3 = param3(i);
        hp_tune_result(i,2) = p2;
        hp_tune_result(i,3) = p3;
        cv_result = crossValidation(k_fold, features, labels, kernel_function, p1, p2, p3);
    elseif exist('param2', 'var')
        p2 = param2(i);
        hp_tune_result(i,2) = p2;
        cv_result = crossValidation(k_fold, features, labels, kernel_function, p1, p2);
    else
        cv_result = crossValidation(k_fold, features, labels, kernel_function, p1);
    end
    for j=4:5
        hp_tune_result(i,j) = cv_result{j-3}; % try to take the first one first
    end
end

