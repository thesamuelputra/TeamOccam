function hp_tune_result = GridSearchCV(k_fold, features, labels, kernel_function, param1, param2, param3)
% GridSearchCV for hyperparameter tuning

length_param = length(param);
hp_tune_result = zeros(length_param, 4); % c, gamma, epsilon, cv_result
for i=1:length_param
    p1 = param1(i);
    hp_tune_result(i,1) = p1;
    if exist('param2', 'var') 
        p2 = param2(i);
        hp_tune_result(i,2) = p2;
        cv_result = crossValidation(k_fold, features, labels, kernel_function, p1, p2);
    elseif exist('param3', 'var')
        p3 = param3(i);
        hp_tune_result(i,3) = p3;
        cv_result = crossValidation(k_fold, features, labels, kernel_function, p1, p2, p3);
    else
        cv_result = crossValidation(k_fold, features, labels, kernel_function, p1);
    end
    hp_tune_result(i,4) = size(cv_result{1}.SupportVectors,1); % try to take the first one first
end
end

