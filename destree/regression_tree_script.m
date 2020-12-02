%% Import data
data = readtable('../datasets/ccu.csv', 'Range', 'G:CY');

%% Clean data and Decision Tree Function
features = table2array(data);
labels = table2array(readtable('../datasets/ccu.csv', 'Range', 'DZ:DZ'));

m = round(size(features,1)*0.80);
features_train = features(1:m,:);
features_test = features(m+1:end,:);
labels_train = labels(1:m);
labels_test = labels(m:end);

headers = data.Properties.VariableNames;
max_depth = 10;
classification = false;

cv_result = CrossValidation(features_train, labels_train, headers, max_depth, classification);
disp('Cross Validation Regression Result')
disp(cv_result)

tree = DecisionTreeLearning(features_train, labels_train, headers, max_depth, classification);
rmse = PredictTreeRegression(tree, labels_train, labels_test);
disp(['RMSE: ' num2str(rmse)])
DrawDecisionTree(tree)