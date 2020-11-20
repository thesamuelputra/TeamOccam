%features = table2array(readtable('./datasets/ccu.csv','Range', 'F:CY'));
%label = table2array(readtable('./datasets/ccu.csv','Range','DZ:DZ'));

function destree = DecisionTreeLearning(features,labels)
[best_attribute, best_threshold] = chooseAttribute(features,targets);
tree.attribute = best_attribute;
tree.threshold = best_threshold;
%if tree.attribute < tree.threshold
%    tree.branch = {}
disp(tree)
end

function [best_attribute,best_threshold] = chooseAttribute(features, targets)
% measures how “good” each attribute (i.e. feature) in the set is.
for i = 1:length(features)
    disp(features(:,i))
end
    % get whole row (features(i,:))
end