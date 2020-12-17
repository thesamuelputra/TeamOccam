function model = poly_r(features, label, q, epsilon)
model = fitrsvm(features, label, 'KernelFunction', 'polynomial', 'PolynomialOrder', q, 'Epsilon', epsilon);
end