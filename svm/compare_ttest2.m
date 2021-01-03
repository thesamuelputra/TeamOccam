destree_cr = 0.68701;
destree_rmse = 23.4506;
ann_rmse = 0.2626;

destree_poly(1,1) = destree_cr;
destree_poly(1,2) = destree_rmse;
destree_poly(2,1) = best_poly_cr;
% destree_poly(2,1) = 0.8960;
destree_poly(2,2) = best_poly_rmse;
% destree_poly(2,2) = 1.5835;

destree_rbf(1,1) = destree_cr;
destree_rbf(1,2) = destree_rmse;
destree_rbf(2,1) = best_rbf_cr;
% destree_rbf(2,1) = 0.9393;
destree_rbf(2,2) = best_rbf_rmse;
% destree_rbf(2,2) = 2.9550;

ann_poly(1,1) = ann_rmse;
ann_poly(2,1) = best_poly_rmse;

ann_rbf(1,1) = ann_rmse;
ann_rbf(2,1) = best_rbf_rmse;

tt2_destree_poly = cell(1,3);
tt2_destree_rbf = cell(1,3);
tt2_ann_poly = cell(1,3);
tt2_ann_rbf = cell(1,3);

[tt2_destree_poly{1}, tt2_destree_poly{2}, tt2_destree_poly{3}] = ttest2(destree_poly(1,:), destree_poly(2,:));
[tt2_destree_rbf{1}, tt2_destree_rbf{2}, tt2_destree_rbf{3}] = ttest2(destree_rbf(1,:), destree_rbf(2,:));

[tt2_ann_poly{1}, tt2_ann_poly{2}, tt2_ann_poly{3}] = ttest2(ann_poly(1,:), ann_poly(2,:));
[tt2_ann_rbf{1}, tt2_ann_rbf{2}, tt2_ann_rbf{3}] = ttest2(ann_rbf(1,:), ann_rbf(2,:));

