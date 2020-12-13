function mdlr_rbf = rbf_r(features, label, c, gamma, epsilon)
    mdlr_rbf = fitrsvm(features, label, 'KernelFunction', 'Gaussian', 'BoxConstraint', c, 'KernelScale', gamma, 'Epsilon', epsilon);
end