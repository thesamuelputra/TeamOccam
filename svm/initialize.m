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
features_r(2006,:) = [];
labels_r = table2array(readtable('../datasets/ccu.csv', 'Range', 'DZ:DZ'));
labels_r(25) = [];


%% Convert multi-class labels_cs to binary (N = 1, SP = -1)
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
%model1 = poly_c(normalize(features_c), normalize(labels_c), 1);
% model2 = poly_c(normalize(features_c), labels_c, 1, 2);
%model4 = poly_c(normalize(features_c), normalize(labels_c), 3);
%model5 = poly_c(normalize(features_c), normalize(labels_c), 4);
%model3 = rbf_c(normalize(features_c), normalize(labels_c), 0.1, 1);

for i = 1:20
    j = 10;
    if i == j
        continue;
    end
%     sv = model2.SupportVectors;
%     fig = figure('Name', 'Model'+string(i));
%     gscatter(normalize(features_c(:,j)),normalize(features_c(:,i)), labels_c);
%     hold on
%     plot(sv(:,j),sv(:,i),'ko','MarkerSize',10)
%     legend(string(model2.ClassNames(1,1)), string(model2.ClassNames(2,1)),'Support Vector')
%     xlabel(string(j));
%     ylabel(string(i));
%     hold off
%     saveas(fig, fullfile('./figures/' +string(j), 'fig'+string(j)+'-' + string(i) + '.png'));
end
mdlr_linear = fitrsvm(normalize(features_r), normalize(labels_r));
mdlc_linear = fitcsvm(normalize(features_c), normalize(labels_c), 'KernelFunction','linear', 'BoxConstraint', 1);
