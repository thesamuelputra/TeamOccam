%% Set up the Import Options and import the data
optsx = spreadsheetImportOptions("NumVariables", 20);
optsy = spreadsheetImportOptions("NumVariables", 1);

% Specify sheet and range
optsx.Sheet = "Data";
optsx.DataRange = "K3:AD2128";
optsx.VariableNamesRange = "K2:AD2";
optsy.Sheet = "Data";
optsy.DataRange = "AT3:AT2128";
optsy.VariableNamesRange = "AT2";


% Specify column names and types
optsx.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
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

n = round(size(array_x,1)*0.80);
array_x_train = array_x(1:n,:);
array_y_train = array_y(1:n);
array_x_test = array_x(n:length(array_x),:);
array_y_test = array_y(n:length(array_y));

headers = x.Properties.VariableNames;
max_depth = 10;
classification = true;

cv_result = CrossValidation(array_x_train, array_y_train, headers, max_depth, classification);
disp('Cross Validation Classifcation Result')
disp(cv_result)

tree = DecisionTreeLearning(array_x_train, array_y_train, headers, max_depth, classification);
[recall, precision] = PredictTreeClassification(tree, array_x_test, array_y_test);
disp(['Recall: ' num2str(recall)])
disp(['Precision: ' num2str(precision)])
DrawDecisionTree(tree)