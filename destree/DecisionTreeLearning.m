% data = readtable('./datasets/ccu.csv', 'Range', 'G:X');
% features = table2array(data);
% label = table2array(readtable('./datasets/ccu.csv', 'Range', 'DZ:DZ'));
% headers = data.Properties.VariableNames;
% max_depth = 10;
% 
% m = round(size(features,1)*0.80);
% idx = randperm(m);
% features_train = features(1:m,:);
% features_test = features(m+1:end,:);
% label_train = label(1:m);
% label_test = label(m:end);

function tree = DecisionTreeLearning(features, label, headers, max_depth)
% fprintf('Calculating tree depth...');
% disp(max_depth);
min_value = 7; % add to parameter?
tree.op = '';
tree.kids = [];
tree.prediction = length(label);

if max_depth ~= 0 && ~isempty(features)
    if (size(features,1) > min_value)
        % may split node into 2 branch
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
        if (~isequal(features1, features) && ~isequal(features2, features))
            features1(:,tree.attribute) = [];
            features2(:,tree.attribute) = [];
            tree.kids{1} = DecisionTreeLearning(features1, label1, headers, max_depth-1);
            tree.kids{2} = DecisionTreeLearning(features2, label2, headers, max_depth-1);
        else
            % leaf node because data are not splitted
            tree.prediction = mean(label);
        end
    else
        % leaf node
        tree.prediction = mean(label);
    end
end

end
