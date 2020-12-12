function mdlc_rbf = rbf_c(features, label, c, gamma)
mdlc_rbf = fitcsvm(features, label, 'KernelFunction', 'Gaussian', 'BoxConstraint', c, 'KernelScale', gamma);
end