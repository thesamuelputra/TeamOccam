%features = table2array(readtable('./datasets/ccu.csv','Range', 'F:CY'));
%features = table2array(readtable('./datasets/ccu.csv','Range','G:O'));
%features = table2array(readtable('./datasets/ccu.csv','Range','G:X'));
%label = table2array(readtable('./datasets/ccu.csv','Range','DZ:DZ'));

function tree = DecisionTreeLearning(features, label)
% look at entropy at the label, not the features?
% if all examples have the same label then return a leaf node with label = the label
% how to determine leaf node?
disp('Calculating tree...');
[best_attribute, best_threshold] = ChooseAttribute(features,label);
tree.attribute = best_attribute;
tree.threshold = best_threshold;
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
tree.features1 = features1;
tree.features2 = features2;
tree.label1 = label1;
tree.label2 = label2;
if (~isempty(features1) && ~isequal(features,features1))
    tree.subtree1 = DecisionTreeLearning(features1, label1);
elseif (~isempty(label1))
    tree.leaf1 = mean(label1);
end
if (~isempty(features2) && ~isequal(features,features2))
    tree.subtree2 = DecisionTreeLearning(features2, label2);
elseif (~isempty(label2))
    tree.leaf2 = mean(label2);
end
end