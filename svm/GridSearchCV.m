function hp_tune_result = GridSearchCV(k_fold, features, labels, kernel_function, param1, param2, param3)
% GridSearchCV for hyperparameter tuning
% param1 = c
% param2 = gamma
% param3 = epsilon

length_c = length(param1);
length_gamma = length(param2);
hp_tune_result = zeros(length_c, 13); % c, gamma, epsilon, cv_result

if length_c == length_gamma
    for i=1:length_c
        c = param1(i);
        gamma = param2(i);
        
        cv_result = crossValidation(k_fold, features, labels, kernel_function, c, gamma);
        hp_tune_result(i,1) = c;
        hp_tune_result(i,2) = gamma;
        for j=4:13
            hp_tune_result(i,j) = cv_result{j-3}; % try to take the first one first
        end
    end
else
    error('Error: parameters should have same length');
end
end

