%features = table2array(readtable('./datasets/ccu.csv','Range', 'F:CY'));
%features = table2array(readtable('./datasets/ccu.csv','Range','G:O'));
%label = table2array(readtable('./datasets/ccu.csv','Range','DZ:DZ'));

function tree = DecisionTreeLearning(features, label, level)
% change level to leaf nodes constraint in if condition
if level <=10
    % look at entropy at the label, not the features?
    % if all examples have the same label then return a leaf node with label = the label
    disp('Calculating tree...');
    [best_attribute, best_threshold] = ChooseAttribute(features,label);
    tree.attribute = features(:,best_attribute);
    tree.threshold = best_threshold;
    features1 = [];
    features2 = [];
    label1 = [];
    label2 = [];
    for i = 1:size(tree.attribute,1)
        if (tree.attribute(i) < tree.threshold)
            features1 = [features1; features(i,:)];
            label1 = [label1; label(i)];
        elseif (tree.attribute(i) >= tree.threshold)
            features2 = [features2; features(i,:)];
            label2 = [label2; label(i)];
        end
    end
    tree.features1 = features1;
    tree.features2 = features2;
    tree.label1 = label1;
    tree.label2 = label2;
    if (~isempty(features1))
        tree.subtree1 = DecisionTreeLearning(features1, label1, level+1);
    end
    if (~isempty(features2))
        tree.subtree2 = DecisionTreeLearning(features2, label2, level+1);
    end
else
    tree = 0;
end
end