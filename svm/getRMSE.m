function rmse = getRMSE(predicted, labels)
sum = 0;
for i=1:size(predicted,1)
    sum = sum + (predicted(i) - labels(i))^2;
end
rmse = sqrt(sum/size(predicted,1));
end

