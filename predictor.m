clear
clc
close all

R = cell(13,1);

R{1} = load('DLSSVM/results_videos_pristine_high.mat');
R{1}.c3dendfile = '.mat';
R{2} = load('DLSSVM/results_videos_blur_high.mat');
R{2}.c3dendfile = '_blur_high.mat';
R{3} = load('DLSSVM/results_videos_blur_medium.mat');
R{3}.c3dendfile = '_blur_medium.mat';
R{4} = load('DLSSVM/results_videos_blur_low.mat');
R{4}.c3dendfile = '_blur_low.mat';
R{5} = load('DLSSVM/results_videos_gaussian_high.mat');
R{5}.c3dendfile = '_gaussian_high.mat';
R{6} = load('DLSSVM/results_videos_gaussian_medium.mat');
R{6}.c3dendfile = '_gaussian_medium.mat';
R{7} = load('DLSSVM/results_videos_gaussian_low.mat');
R{7}.c3dendfile = '_gaussian_low.mat';
R{8} = load('DLSSVM/results_videos_mpeg4_high.mat');
R{8}.c3dendfile = '_high_MPEG4.mat';
R{9} = load('DLSSVM/results_videos_mpeg4_medium.mat');
R{9}.c3dendfile = '_medium_MPEG4.mat';
R{10} = load('DLSSVM/results_videos_mpeg4_low.mat');
R{10}.c3dendfile = '_low_MPEG4.mat';
R{11} = load('DLSSVM/results_videos_saltpepper_high.mat');
R{11}.c3dendfile = '_sp_high.mat';
R{12} = load('DLSSVM/results_videos_saltpepper_medium.mat');
R{12}.c3dendfile = '_sp_medium.mat';
R{13} = load('DLSSVM/results_videos_saltpepper_low.mat');
R{13}.c3dendfile = '_sp_low.mat';
nR = length(R);
foldername = 'C3D';
nvideos = 70;
y = zeros(nR*nvideos,1);
X = zeros(nR*nvideos,4096);
for jj = 1:nR
    for ii = 1:nvideos
        try
            F = load([foldername '/video' num2str(ii) R{jj}.c3dendfile]);
            Fv = mean(F.Feature_vect,1);
            X(nvideos*(jj-1)+ii,:) = Fv;
            y(nvideos*(jj-1)+ii) = R{jj}.auc_per(ii);
        catch
            X(nvideos*(jj-1)+ii,:) = nan;
            y(nvideos*(jj-1)+ii) = nan;
        end
    end
end


% Wr = randn(4096,3);
ind = 1:length(y);
ind(isnan(y))=[];
X = X(ind,:);
y = y(ind);
nobj = length(y);
ntr = round(0.7*nobj);
ind = randperm(nobj);
indtrain = ind(1:ntr);
indtest = ind(ntr+1:end);
Xr_train = X(indtrain,:); % *Wr;
Xr_test = X(indtest,:); % *Wr;
Yr_train = y(indtrain); % *Wr;
Yr_test = y(indtest); % *Wr;
[~,strain] = sort(Yr_train,'ascend');
[~,stest] = sort(Yr_test,'ascend');
Xr_train = Xr_train(strain,:);
Xr_test = Xr_test(stest,:);
Yr_train = Yr_train(strain);
Yr_test = Yr_test(stest);
indtrain = indtrain(strain);
indtest = indtest(stest);
% mdl = fitlm(Xr_train,y(indtrain));
mdl = fitrsvm(Xr_train,y(indtrain),'KernelFunction','rbf','KernelScale',50);
ypred = predict(mdl,Xr_test);
ypredtrain = predict(mdl,Xr_train);

figure
subplot(1,2,1)
hold on
plot(ypredtrain,'r')
plot(y(indtrain))
hold off
legend({'Prediction','AUC DLSSVM'},'Location','southeast')
title('Training')

subplot(1,2,2)
hold on
plot(ypred,'r')
plot(y(indtest))
hold off
legend({'Prediction','AUC DLSSVM'},'Location','southeast')
title('Test')

ytest = y(indtest);

%  Fit svm



