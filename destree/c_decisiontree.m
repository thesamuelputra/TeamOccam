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
% t = sum(array_y);
%val_n = sum(array_y(:)==1);
%val_s = sum(array_y(:)==-1);
% 
% disp(t)
% disp(val_n)
% disp(val_s)

n = round(size(array_x,1)*0.80);
array_x_train = array_x(1:n,:);
array_y_train = array_y(1:n);
array_x_test = array_x(n:length(array_x),:);
array_y_test = array_y(n:length(array_y));

headers = x.Properties.VariableNames;
max_depth = 10;

tree = DecisionTreeLearning(array_x_train, array_y_train, headers, max_depth);
k_tree = TreeToKFold(array_x_train, array_y_train, headers, max_depth);
% DrawDecisionTree(tree);

function tree = DecisionTreeLearning(features, label, headers, max_depth)
% fprintf('Calculating tree depth...');
% disp(max_depth);
min_value = 7;
tree.op = '';
tree.kids = [];
tree.prediction = 5;

if max_depth ~= 0 && ~isempty(features) && (size(features,1) > min_value)
    % may split node into 2 branch
    [best_attribute, best_threshold] = ChooseAttribute(features, label);
    tree.attribute = best_attribute;
    tree.threshold = best_threshold;
    tree.op = headers{tree.attribute};
    features1 = [];
    features2 = [];
    label1 = [];
    label2 = [];
    for i = 1:size(features(:,tree.attribute),1)
        if (features(i,tree.attribute) < tree.threshold)
            features1 = [features1; features(i,:)];
            label1 = [label1; label(i)];
        elseif (features(i, tree.attribute) >= tree.threshold)
            features2 = [features2; features(i,:)];
            label2 = [label2; label(i)];
        end
    end
    if (~isequal(features1, features) && ~isequal(features2, features))
        %features1(:,tree.attribute) = [];
        %features2(:,tree.attribute) = [];
        tree.kids{1} = DecisionTreeLearning(features1, label1, headers, max_depth-1);
        tree.kids{2} = DecisionTreeLearning(features2, label2, headers, max_depth-1);
    else
        % leaf node because data are not splitted
        tree.prediction = getMajorityValue(label);
    end
else
    % leaf node
    tree.prediction = getMajorityValue(label);
end

end

function [best_attribute, best_threshold, attributes] = ChooseAttribute(features, label)
% measures how “good” each attribute (i.e. feature) in the set is.
attributes = zeros(2,size(features,2));
for i = 1:size(features,2)
    [attributes(1,i), attributes(2,i)] = getThreshold(features(:,i), label);
end
best_attribute = find(attributes(1,:)==min(attributes(1,:)),1,'first'); % return the column index
best_threshold = attributes(2, best_attribute);
fprintf('Best Attribute...');
disp(best_attribute);
fprintf('Best Threshold...');
disp(best_threshold);
end

function [min_gini,threshold] = getThreshold(col, array_y)
% Merge then sort then split
table = [col, array_y];
table = sortrows(table, 1);
col = table(:,1);
array_y = table(:,2);

min_gini = 1/0;
% gini = 1 - probno^2 - probyes^2;
threshold = col(1);

for i = 1:(length(col)-1)
%   for calc gini in continuous data 
    avg_weight = (col(i) + col(i+1)) / 2;
    prob_l_pos = sum(array_y(1:i) == 1);
    prob_l_neg = sum(array_y(1:i) == -1);
    prob_l_total = length(array_y(1:i));
    gini_l = 1 - (prob_l_pos/(prob_l_total))^2 - (prob_l_neg/(prob_l_total))^2;
    prob_r_pos = sum(array_y((i+1):length(col)) == 1);
    prob_r_neg = sum(array_y((i+1):length(col)) == -1);
    prob_r_total = length(array_y((i+1):length(col)));
    gini_r = 1 - (prob_r_pos/(prob_r_total))^2 - (prob_r_neg/(prob_r_total))^2;
    total_gini = (prob_l_total/(prob_l_total+prob_r_total))*gini_l + (prob_r_total/(prob_l_total+prob_r_total))*gini_r;
    if (total_gini < min_gini)
        min_gini = total_gini;
        % choose the row with minimum gini as the threshold
        threshold = avg_weight;
    end
end
end

function majority = getMajorityValue(label)
    majority = sum(label(:)==1) > sum(label(:)==-1);
end

function [k_tree, cv_result] = TreeToKFold(features_train, label_train, headers, max_depth)
k = 10;
cv_result = zeros(1,k);
k_tree = cell(1,k);
rowsize = size(features_train,1);
rand = randperm(rowsize);
shuffled_features = features_train(rand,:);
shuffled_label = label_train(rand,:);
start_index = 1;
for i=1:k
    end_index = round((rowsize/k)*i);
    ftest = shuffled_features(start_index:end_index,:);
    ltest = shuffled_label(start_index:end_index);
    ftrain = features_train;
    ltrain = label_train;
    ftrain(start_index:end_index,:) = [];
    ltrain(start_index:end_index) = [];
    k_tree{i} = DecisionTreeLearning(ftrain, ltrain, headers, max_depth);
    cv_result(1,i) = PredictTreeClassification(k_tree{i}, ftest, ltest);
    start_index = end_index;
end
end