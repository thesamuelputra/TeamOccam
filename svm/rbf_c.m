function mdlc_rbf = rbf_c(features, labels, c, gamma)
    mdlc_rbf = fitcsvm(features, labels, 'KernelFunction', 'Gaussian', 'BoxConstraint', c, 'KernelScale', gamma);
end