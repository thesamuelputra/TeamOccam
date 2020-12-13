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
label_c = readtable("../datasets/CTG.xls", optsy);


%% Convert multi-class label_cs to binary (N = 1, SP = -1)
for i = 1:height(label_c)
    if  label_c.NSP(i) == 2 || label_c.NSP(i) == 3
        label_c.NSP(i) = -1;
    end
end

%% Clear temporary variables
clear optsx optsy i;
features_c = table2array(features_c);
label_c = table2array(label_c);

%% SVM

% mdlc_linear = fitcsvm(features_c, label_c, 'KernelFunction','linear', 'BoxConstraint', 1);
c = [1,2,3,4,5,6,7,8,9,10];
gamma = [1,2,3,4,5,6,7,8,9,10];



model = polynomial_classification(normalize(features_c), normalize(label_c));