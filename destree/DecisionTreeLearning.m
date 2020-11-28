%features = table2array(readtable('./datasets/ccu.csv','Range', 'F:CY'));
%features = table2array(readtable('./datasets/ccu.csv','Range','G:O'));
% features = table2array(readtable('./datasets/ccu.csv','Range','G:X'));
% label = table2array(readtable('./datasets/ccu.csv','Range','DZ:DZ'));

function tree = DecisionTreeLearning(features, label, max_depth)
% look at entropy at the label, not the features?
% if all examples have the same label then return a leaf node with label = the label
% how to determine leaf node?
% 1. Max Depth
% 2. Minimum value in features1/features2
% 3. Use Mean to predict
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
tree.subtree1 = Branching(features1, label1, max_depth);
tree.subtree2 = Branching(features2, label2, max_depth);
end

function subtree = Branching(features, label, max_depth)
subtree = 0;
min_value = 10;
if max_depth ~= 0
    if (~isempty(features))
        if (size(features,1) > min_value)
            subtree = DecisionTreeLearning(features, label, max_depth-1);
        else
            subtree = mean(features);
        end
    end
end
end