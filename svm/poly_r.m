function model = poly_r(features, labels, q, epsilon)
model = fitrsvm(features, labels, 'KernelFunction', 'polynomial', 'PolynomialOrder', q, 'Epsilon', epsilon);
end