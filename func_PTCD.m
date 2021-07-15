function result = func_PTCD(hsi_t1,hsi_t2,win_size,step)
%FUNC_PTCD Patch Tensor-based Change Detection
%
% Fucction Usage:
%   [result] = func_TensorBlockCD(hsi_t1,hsi_t2,win_size,step)
%   INPUTS:
%       hsi_t1 -> the 3D hyperspectral imagery(rows x cols x bands) at t1 time 
%       hsi_t2 -> the 3D hyperspectral imagery(rows x cols x bands) at t2 time 
%     win_size -> the size of Block with the size of [size_rows,size_cols]
%         step -> the step between blocks with the size of [size_rows,size_cols]
%   OUTPUT:
%     result -> the change detection result.
%
% Author: Zephyr Hou
% Time: 2020-12-31
%
%All Rights Reserved.
%Contact Information: zephyrhours@gmail.com

%% ========================= Main Function ========================
[rows,cols,bands] = size(hsi_t1);
model=3; % the mode of unfold 

%% step1: Image Segmentation Block Processing
Block_t1=[];
Block_t2=[];
ind_r=win_size(1):step(1):rows; 
ind_c=win_size(2):step(2):cols;
ind_r=ind_r-win_size(1)+1;     % first index of patch (rows)
ind_c=ind_c-win_size(2)+1;     % first index of patch (cols)

nums_r=length(ind_r);
nums_c=length(ind_c);

k=1;
for i =1:nums_r
    for j = 1:nums_c
        ii=ind_r(i);
        jj=ind_c(j);

        tempBlock_t1=hsi_t1(ii:win_size(1)+ii-1,jj:win_size(2)+jj-1,:);
        tempBlock_t2=hsi_t2(ii:win_size(1)+ii-1,jj:win_size(2)+jj-1,:);
        
        Block_t1(:,:,k)=reshape(tempBlock_t1,win_size(1)*win_size(2),bands);      
        Block_t2(:,:,k)=reshape(tempBlock_t2,win_size(1)*win_size(2),bands);  
 
        k=k+1;
    end
end



%% step2: Tensor Reconstruction
[H,W,Dim]=size(Block_t1);
PC=min(min(H,W),Dim);
%% reconstruction for hsi_t1
Ten_Block=tensor(Block_t1);
[~,~,core,U]=tucker_als(Ten_Block,[H,W,Dim]);
core_new=core(1:PC,1:PC,1:PC);
U_new{1}=U{1}(:,1:PC);
U_new{2}=U{2}(:,1:PC);
U_new{3}=U{3}(:,1:PC);

new_Block = ttensor(core_new, U_new);
unfold_model = tenmat(new_Block,model);
unfold_model = double(unfold_model);
new_Block=reshape(unfold_model',H,W,Dim); 

%% 
newBlock={};
for kk = 1:Dim
    tmpnewBlock=new_Block(:,:,kk);
    newBlock{kk}=reshape(tmpnewBlock,win_size(1),win_size(2),bands);
end
new_hsi=hsi_t1;
k=1;
for i = 1:nums_r
    for j = 1:nums_c
        ii=ind_r(i);
        jj=ind_c(j);
        new_hsi(ii:win_size(1)+ii-1,jj:win_size(2)+jj-1,:)=newBlock{k};
        k=k+1;
    end
end            
newhsi_t1=new_hsi;
%% reconstruction for hsi_t2
Ten_Block=tensor(Block_t2);
[~,~,core,U]=tucker_als(Ten_Block,[H,W,Dim]);
core_new=core(1:PC,1:PC,1:PC);
U_new{1}=U{1}(:,1:PC);
U_new{2}=U{2}(:,1:PC);
U_new{3}=U{3}(:,1:PC);

new_Block = ttensor(core_new, U_new);
unfold_model = tenmat(new_Block,model);
unfold_model = double(unfold_model);
new_Block=reshape(unfold_model',H,W,Dim); 
%%
newBlock={};
for k = 1:Dim
    tmpnewBlock=new_Block(:,:,k);
    newBlock{k}=reshape(tmpnewBlock,win_size(1),win_size(2),bands);
end
new_hsi=hsi_t2;
k=1;
for i = 1:nums_r
    for j = 1:nums_c
        ii=ind_r(i);
        jj=ind_c(j);
        new_hsi(ii:win_size(1)+ii-1,jj:win_size(2)+jj-1,:)=newBlock{k};
        k=k+1;
    end
end            
newhsi_t2=new_hsi;

%% step3: New Designed Detector for Hyperspectral Change Detection
% Revised Local Abosulte Distance(RLAD)
% eight-neighborhood pixels
win_out=3;
win_in=1;

[rows, cols, bands] = size(newhsi_t1);
result = zeros(rows, cols);
t = fix(win_out/2);
t1 = fix(win_in/2);
M = win_out^2;
%% adaptive boundary filling
DataTest1 = zeros(rows+2*t,cols+2*t, bands);
DataTest1(t+1:rows+t, t+1:cols+t, :) = newhsi_t1;
DataTest1(t+1:rows+t, 1:t, :) = newhsi_t1(:, t:-1:1, :);
DataTest1(t+1:rows+t, t+cols+1:cols+2*t, :) = newhsi_t1(:, cols:-1:cols-t+1, :);
DataTest1(1:t, :, :) = DataTest1(2*t:-1:(t+1), :, :);
DataTest1(t+rows+1:rows+2*t, :, :) = DataTest1(t+rows:-1:(rows+1), :, :);

DataTest2 = zeros(rows+2*t,cols+2*t, bands);
DataTest2(t+1:rows+t, t+1:cols+t, :) = newhsi_t2;
DataTest2(t+1:rows+t, 1:t, :) = newhsi_t2(:, t:-1:1, :);
DataTest2(t+1:rows+t, t+cols+1:cols+2*t, :) = newhsi_t2(:, cols:-1:cols-t+1, :);
DataTest2(1:t, :, :) = DataTest2(2*t:-1:(t+1), :, :);
DataTest2(t+rows+1:rows+2*t, :, :) = DataTest2(t+rows:-1:(rows+1), :, :);
%%  change detection
for i = t+1:cols+t 
    for j = t+1:rows+t
        block1 = DataTest1(j-t: j+t, i-t: i+t, :);
        y1 = squeeze(DataTest1(j, i, :));   % num_dim x 1
        block1(t-t1+1:t+t1+1, t-t1+1:t+t1+1, :) = NaN;
        block1 = reshape(block1, M, bands);
        block1(isnan(block1(:, 1)), :) = [];
        H1 = block1';  % num_dim x num_sam
		
		block2 = DataTest2(j-t: j+t, i-t: i+t, :);
        y2 = squeeze(DataTest2(j, i, :));   % num_dim x 1
        block2(t-t1+1:t+t1+1, t-t1+1:t+t1+1, :) = NaN;
        block2 = reshape(block2, M, bands);
        block2(isnan(block2(:, 1)), :) = [];
        H2 = block2';  % num_dim x num_sam	
       %% designed new detector(RLAD)
        tempD=sqrt(sum((H2.^2-H1.^2).^2));          % 1 x num_sam   
        w=atan(((y1'*y2)/(norm(y1)*norm(y2)))^2);   % Tan
       %%
        result(j-t, i-t) =sum(tempD)*w;    
    end
end

end

