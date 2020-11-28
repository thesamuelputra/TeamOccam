% Use this function as root on completion

% function tree = DecisionTreeLearning()
% data = readtable('./datasets/ccu.csv', 'Range', 'G:X');
% features = table2array(data);
% label = table2array(readtable('./datasets/ccu.csv', 'Range', 'DZ:DZ'));
% headers = data.Properties.VariableNames;
% max_depth = 10;
% tree = CreateTree(features, label, headers, max_depth);
% end

% Change to CreateTree on completion
% function sometimes return empty kid
function tree = DecisionTreeLearning(features, label, headers, max_depth)
% fprintf('Calculating tree depth...');
% disp(max_depth);
[best_attribute, best_threshold] = ChooseAttribute(features);
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
% tree.features1 = features1;
% tree.features2 = features2;
% tree.label1 = label1;
% tree.label2 = label2;
tree.kids = [];
tree.prediction = NaN;
tree = Branching(tree, 1, features, features1, label1, headers, max_depth);
tree = Branching(tree, 2, features, features2, label2, headers, max_depth);
end

function tree = Branching(tree, kid_index, root_features, features, label, headers, max_depth)
min_value = 20;
if max_depth ~= 0
    if (~isempty(features))
        if (size(features,1) > min_value || ~isequal(root_features, features))
            tree.kids{kid_index} = DecisionTreeLearning(features, label, headers, max_depth-1);
        else
            tree.prediction = mean(features);
        end
    end
end
end