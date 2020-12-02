========= READING DATA & VARIABLES (REGRESSION) =========

data = readtable('../datasets/ccu.csv', 'Range', 'G:CY');
features = table2array(data);
labels = table2array(readtable('../datasets/ccu.csv', 'Range', 'DZ:DZ'));
headers = data.Properties.VariableNames;
max_depth = 10;

m = round(size(features,1)*0.80);
features_train = features(1:m,:);
features_test = features(m+1:end,:);
label_train = labels(1:m);
label_test = labels(m:end);

CALL FUNCTION

[k_tree, cv_result] = TreeToKFold(features_train, label_train, headers, max_depth);

========= READING DATA & VARIABLES (CLASSIFICATION) =========
%% Set up the Import Options and import the data
optsx = spreadsheetImportOptions("NumVariables", 21);
optsy = spreadsheetImportOptions("NumVariables", 1);

% Specify sheet and range
optsx.Sheet = "Data";
optsx.DataRange = "K3:AE2128";
optsx.VariableNamesRange = "K2:AE2";
optsy.Sheet = "Data";
optsy.DataRange = "AT3:AT2128";
optsy.VariableNamesRange = "AT2";


% Specify column names and types
optsx.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
optsy.VariableTypes = "double";

% Import the data
x = readtable("../datasets/CTG.xls", optsx);
y = readtable("../datasets/CTG.xls", optsy);

%% Clear temporary variables
clear optsx optsy;

%% Convert multi-class labels to binary (N = 1, SP = -1)
for i = 1:height(y)
    if  y.NSP(i) == 2 || y.NSP(i) == 3
        y.NSP(i) = -1;
    end
end

%% Decision Tree Function
array_x = table2array(x);
array_y = table2array(y);

CALL FUNCTION

tree = DecisionTreeLearning(array_x, array_y, x.Properties.VariableNames, 10);
DrawDecisionTree(tree);
