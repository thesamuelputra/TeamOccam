function model = poly_c(features, label, q)
model = fitcsvm(features, label, 'KernelFunction', 'polynomial', 'PolynomialOrder', q);
end