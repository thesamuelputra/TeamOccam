function rmse = rmse(predicted, labels)
sum = 0;
for j=1:size(predicted,1)
    sum = sum + (predicted(j) - labels(j))^2;
end
rmse = sqrt(sum/size(predicted,1));
end

