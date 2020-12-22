function fmeasure = getClassRate(predicted,labels)
tp = 0;
fp = 0;
fn = 0;
tn = 0;
for i=1:size(predicted,1)
    if predicted(i) == 1 && labels(i) == 1
        tp = tp + 1;
    elseif predicted(i) == 1 && labels(i) == -1
        fp = fp + 1;
    elseif predicted(i) == -1 && labels(i) == 1
        fn = fn + 1;
    elseif predicted(i) == -1 && labels(i) == -1
        tn = tn + 1;
    end
end
precision = tp/(tp + fp);
recall = tp/(tp + fn);
fmeasure = (2 * precision * recall)/(precision + recall);
end

