function normalized_data = Normalize(data)
normalized_data = zeros(size(data,1),size(data,2));
for i=1:size(data,2)
    m = mean(data(:,i));
    s = std(data(:,i));
    for j=1:size(data(:,1))
        col = data(j,i);
        normalized_data(j,i) = NormalizeColumn(col,m,s);
%         fprintf('mean = ');
%         disp(m);
%         fprintf('std = ');
%         disp(s);
%         fprintf('col = ');
%         disp(col);
%         fprintf('normalized_col = ');
%         disp(normalized_data(j,i));
    end
end
end

function normalized_column = NormalizeColumn(col, mean, std)
normalized_column = (col-mean)/std;
end