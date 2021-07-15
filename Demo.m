% This is a demo for PTCD method
%% Load dataset
clc;clear;close all;
addpath(genpath(pwd));
load('CDdataset_Hermiston.mat');
win_size=[3,3];
step=win_size;

[rows,cols,bands] = size(hsi_t1);
label_value=reshape(hsi_gt,1,rows*cols);

%% PTCD
tic
R0 = func_PTCD(hsi_t1,hsi_t2,win_size,step);
t0=toc;

R0value = reshape(R0,1,rows*cols);
[FA0,PD0] = perfcurve(label_value,R0value,'1') ;
AUC0=-sum((FA0(1:end-1)-FA0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
disp(['PTCD:    ',num2str(AUC0)])

%% AUC Value && Execution Time
clc;
disp('-------------------------------------------------------------------')
disp('PTCD')
disp(['AUC:     ',num2str(AUC0),'          Time:     ',num2str(t0)])
disp('-------------------------------------------------------------------')
%% ROCÇúÏß»æÖÆ
figure;
plot(FA0, PD0, 'k-', 'LineWidth', 2);  hold on
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('PTCD','location','southeast')
















