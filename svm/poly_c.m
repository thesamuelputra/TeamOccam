function model = poly_c(features, labels, c, q)
model = fitcsvm(features, labels, 'KernelFunction', 'polynomial', 'KernelScale', c, 'PolynomialOrder', q);
end