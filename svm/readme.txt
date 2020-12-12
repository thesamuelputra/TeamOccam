REGRESSION
Mdl = fitrsvm(X,Y, Name, Value);

CLASSIFICATION
Mdl = fitcsvm(X,Y, 'KernelFunction','linear', 'BoxConstraint',1);