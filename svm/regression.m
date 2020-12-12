features_r = table2array(readtable('../datasets/ccu.csv', 'Range', 'G:CY'));
label_r = table2array(readtable('../datasets/ccu.csv', 'Range', 'DZ:DZ'));

mdlr_linear = fitrsvm(features_r, label_r, 'KernelFunction', 'linear');