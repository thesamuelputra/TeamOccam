function [recall, precision] = PredictTreeClassification(tree, features, label)
row_size = size(features,1);
tree_prediction = zeros(row_size,1);
confusion_matrix = zeros(2,2);
% apply the destree to each row, get the leaf
for i=1:row_size
    tree_prediction(i) = GetLeafValue(tree, features, i);
    if tree_prediction(i) == 0
        tree_prediction(i) = -1;
    end
    if tree_prediction(i) == label(i) 
        if label(i) == 1
            confusion_matrix(1,1) = confusion_matrix(1,1) + 1;
        elseif label(i) == -1
            confusion_matrix(2,2) = confusion_matrix(2,2) + 1;
        end
    elseif tree_prediction(i) ~= label(i)
        if tree_prediction(i) == 1
            confusion_matrix(1,2) = confusion_matrix(1,2) + 1;
        elseif tree_prediction(i) == -1
            confusion_matrix(2,1) = confusion_matrix(2,1) + 1;
        end
    end
end
precision = confusion_matrix(1,1) / (confusion_matrix(1,1) + confusion_matrix(1,2));
recall = confusion_matrix(1,1) / (confusion_matrix(1,1) + confusion_matrix(2,1));
end
