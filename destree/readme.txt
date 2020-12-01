READING DATA & VARIABLES (REGRESSION)

data = readtable('./datasets/ccu.csv', 'Range', 'G:X');
features = table2array(data);
label = table2array(readtable('./datasets/ccu.csv', 'Range', 'DZ:DZ'));
headers = data.Properties.VariableNames;
max_depth = 10;

m = round(size(features,1)*0.80);
idx = randperm(m);
features_train = features(1:m,:);
features_test = features(m+1:end,:);
label_train = label(1:m);
label_test = label(m:end);

CALL FUNCTION

[k_tree, cv_result] = TreeToKFold(features_train, label_train, headers, max_depth);