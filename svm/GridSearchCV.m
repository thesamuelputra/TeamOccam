function hp_tune_result = gridSearchCV(k_fold, features, labels, kernel_function, param1, param2, param3)
% GridSearchCV for hyperparameter tuning
length_param = length(param1);
hp_tune_result = cell(length_param, 7); % c, gamma/poly, epsilon, cv_result
hp_tune_result{6,1} = 'PARAM1';
hp_tune_result{6,2} = 'PARAM2';
hp_tune_result{6,3} = 'PARAM3';
hp_tune_result{6,4} = 'ACCURACY';
hp_tune_result{6,5} = 'BEST MODEL';
hp_tune_result{6,6} = 'PREDICTION';
for i=1:length_param
    p1 = param1(i);
    hp_tune_result{i,1} = p1;
    if exist('param3', 'var')
        p2 = param2(i);
        p3 = param3(i);
        hp_tune_result{i,2} = p2;
        hp_tune_result{i,3} = p3;
        [cv_result, best_mdl] = crossValidation(k_fold, features, labels, kernel_function, p1, p2, p3);
    elseif exist('param2', 'var')
        p2 = param2(i);
        hp_tune_result{i,2} = p2;
        [cv_result, best_mdl] = crossValidation(k_fold, features, labels, kernel_function, p1, p2);
    else
        [cv_result, best_mdl] = crossValidation(k_fold, features, labels, kernel_function, p1);
    end
    hp_tune_result{i,4} = mean(cv_result);
    hp_tune_result{i,5} = best_mdl;
    hp_tune_result{i,6} = predict(best_mdl, features);
end

