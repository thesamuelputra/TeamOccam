function model = poly_c(features, labels, q)
model = fitcsvm(features, labels, 'KernelFunction', 'polynomial', 'PolynomialOrder', q);
end