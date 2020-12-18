function model = poly_r(features, labels, c, q, epsilon)
model = fitrsvm(features, labels, 'KernelFunction', 'polynomial', 'KernelScale', c, 'PolynomialOrder', q, 'Epsilon', epsilon);
end