function model = polynomial_classification(x, y)
model = fitcsvm(x,y, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);
end