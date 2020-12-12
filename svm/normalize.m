function normalized_data = normalize(data)
normalized_data = zeros(size(data,1),size(data,2));
for i=1:size(data,2)
    m = mean(data(:,i));
    s = std(data(:,i));
    for j=1:size(data(:,1))
        col = data(j,i);
        normalized_data(j,i) = normalizeColumn(col,m,s);
    end
end
end

function normalized_column = normalizeColumn(col, mean, std)
normalized_column = (col-mean)/std;
end 