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

features_r = table2array(readtable('../datasets/ccu.csv', 'Range', 'G:CY'));
label_r = table2array(readtable('../datasets/ccu.csv', 'Range', 'DZ:DZ'));


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
model1 = poly_c(normalize(features_c), normalize(label_c), 1);
model2 = poly_c(normalize(features_c), normalize(label_c), 2);
model4 = poly_c(normalize(features_c), normalize(label_c), 3);
model5 = poly_c(normalize(features_c), normalize(label_c), 4);
model3 = rbf_c(normalize(features_c), normalize(label_c), 0.1, 1);

for i = 2:20
    j = 1;
    sv = model2.SupportVectors;
    fig = figure('Name', 'Model'+string(i));
    gscatter(normalize(features_c(:,j)),normalize(features_c(:,i)),normalize(label_c));
    hold on
    plot(sv(:,j),sv(:,i),'ko','MarkerSize',10)
    legend(string(j),string(i),'Support Vector')
    hold off
    saveas(fig, fullfile('C:\Users\david\Downloads\TeamOccam\svm\figures\' +string(j), 'fig'+string(j)+'-' + string(i)), 'png');
end
% mdlr_linear = fitrsvm(features_r, label_r, 'KernelFunction', 'linear');
% mdlc_linear = fitcsvm(features_c, label_c, 'KernelFunction','linear', 'BoxConstraint', 1);
