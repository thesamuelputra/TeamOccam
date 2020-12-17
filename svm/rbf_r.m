function mdlr_rbf = rbf_r(features, labels, c, gamma, epsilon)
    mdlr_rbf = fitrsvm(features, labels, 'KernelFunction', 'Gaussian', 'BoxConstraint', c, 'KernelScale', gamma, 'Epsilon', epsilon);
end