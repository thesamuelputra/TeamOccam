function tree = DecisionTreeLearning(features, label, headers, max_depth, classification)
min_value = 20; % add to parameter?
tree.op = '';
tree.kids = [];
tree.prediction = 'not leaf node';

if max_depth ~= 0 && ~isempty(features) && (size(features,1) > min_value)
    % may split node into 2 branch
    if classification
        [best_attribute, best_threshold] = ChooseAttributeClassification(features, label);
    else
        [best_attribute, best_threshold] = ChooseAttributeRegression(features, label);
    end
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
        if ~classification
            features1(:,tree.attribute) = NaN;
            features2(:,tree.attribute) = NaN;
        end
        tree.kids{1} = DecisionTreeLearning(features1, label1, headers, max_depth-1, classification);
        tree.kids{2} = DecisionTreeLearning(features2, label2, headers, max_depth-1, classification);
    else
        % leaf node because data are not splitted
        tree.prediction = leafNode(label, classification);
    end
else
    % leaf node
    tree.prediction = leafNode(label, classification);
end

end

function prediction = leafNode(label, classification)
    if classification
        prediction = getMajorityValue(label);
    else
        if ~isnan(mean(label))
            prediction = mean(label);
        else
            disp(label)
            prediction = ['nan value: ' size(label)];
        end
    end
end

function majority = getMajorityValue(label)
    majority = sum(label(:)==1) > sum(label(:)==-1);
end
