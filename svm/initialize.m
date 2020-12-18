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
features_c = readtable("../datasets/CTG.xls", optsx);
labels_c = readtable("../datasets/CTG.xls", optsy);

features_r = table2array(readtable('../datasets/ccu.csv', 'Range', 'G:CY'));
features_r(:,25) = [];
labels_r = table2array(readtable('../datasets/ccu.csv', 'Range', 'DZ:DZ'));


%% Convert multi-class label_cs to binary (N = 1, SP = -1)
for i = 1:height(labels_c)
    if  labels_c.NSP(i) == 2 || labels_c.NSP(i) == 3
        labels_c.NSP(i) = -1;
    end
end

%% Clear temporary variables
clear optsx optsy i;
features_c = table2array(features_c);
labels_c = table2array(labels_c);

%% SVM

% mdlr_linear = fitrsvm(features_r, labels_r, 'KernelFunction', 'linear');
% mdlc_linear = fitcsvm(features_c, labels_c, 'KernelFunction','linear', 'BoxConstraint', 1);
