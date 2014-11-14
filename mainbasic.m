
% Animesh Karmakar 
%-----------------------------------------------------------------------------------------------------------------------


clear all;
clc;
load('dataFile.mat');
X=X';
y=y';
disp('Reading data done');

load('testFile.mat');
Xtest=Xtest';
X=[X;Xtest];

% PCA Starting
initAvg = mean(X);
X = X - repmat(initAvg, size(X,1), 1);
initRan = sqrt( mean(X.^2) );
X = X ./ repmat(initRan, size(X,1), 1);
disp('normalization done');
X1=X(:,1:3000);
X2=X(:,3001:6000);
X3=X(:,6001:9000);
X4=X(:,6001:9000);
X5=X(:,9001:12000);
X6=X(:,12001:15001);
X7=X(:,15001:18000);
X8=X(:,18001:21000);
X9=X(:,21001:24000);
X10=X(:,24001:end);
[X1, U1, error1, ~] = pca(X1,0.999);
disp ('pca1 done');
[X2, U2, error2, ~] = pca(X2,0.999);
disp ('pca2 done');
[X3, U3, error3, ~] = pca(X3,0.999);
disp ('pca3 done');
[X4, U4, error4, ~] = pca(X4,0.999);
disp ('pca3 done');
[X5, U5, error5, ~] = pca(X5,0.999);
disp ('pca3 done');
[X6, U6, error6, ~] = pca(X6,0.999);
disp ('pca3 done');
[X7, U7, error7, ~] = pca(X7,0.999);
disp ('pca3 done');
[X8, U8, error8, ~] = pca(X8,0.999);
disp ('pca3 done');
[X9, U9, error9, ~] = pca(X9,0.999);
disp ('pca3 done');
[X10, U10, error10, ~] = pca(X10,0.999);
disp ('pca3 done');

X=[X1 X2 X3 X4 X5 X6 X7 X8 X9 X10];

[X, U, error, ~] = pca(X,0.9999);
disp ('pca done');

% Normalizing:
ran1 = sqrt( var(X) );
X = X ./ repmat(ran1, size(X,1), 1);


Xtest=X(801:end,:);
X=X(1:800,:);

% startYear = 2005;
% endYear = 2012;
% prevYears = 60;
pos = randperm(size(X,1));
X = X(pos,:); 
y= y(pos,:);
input_layer_size = size(X,2);
hidden_layer_size = 400;
num_labels = size(X,2);
lambda= 0.1;
n=0;e=0;
%y=y*100;
origX=X(851:end,:);
origY=y(851:end,:);
%X=X(1:600,:);
%y=y(1:600,:);


disp ('starting sae 1 ');
% sae = saesetup([size(X,2) 400]);
% sae.ae{1}.activation_function       = 'tanh_opt';
% sae.ae{1}.learningRate              = 0.01;
% sae.ae{1}.scaling_learningRate      = 1.0;
% sae.ae{1}.inputZeroMaskedFraction   = 0.02;
% sae.ae{1}.output                    = 'tanh_opt';
% 
%                   

opts.numepochs =   50;
opts.batchsize = 50;
opts.plot = 1;
% sae = saetrain(sae,X , opts);
% visualize(sae.ae{1}.W{1}(:,2:end)');
% 
% sae.ae{1}=nnff(sae.ae{1},X,X);
% features1= sae.ae{1}.a{2};

% sae.ae{2}=nnff(sae.ae{2},features1(:,2:end),features1(:,2:end));
% features2= sae.ae{2}.a{2};
% 
% sae.ae{3}=nnff(sae.ae{3},features2(:,2:end),features2(:,2:end));
% features3= sae.ae{3}.a{2};
%   disp ('starting nn1 ');
% nn1=nnsetup([400 100 size(y,2)]);
% nn1.activation_function       = 'tanh_opt';
% nn1.learningRate              = 0.02;
% nn1.scaling_learningRate      = 1.0;
% nn1.inputZeroMaskedFraction   = 0.0;
% nn1.output                    = 'sigm';
% opts.plot                     =1;
% opts.numepochs =   120;
% nn1.weightPenaltyL2           = 0.01;            %  L2 regularization
% nn1=nntrain(nn1,features1(:,2:end),y,opts);
% disp ('starting nn  ');
% Final Training:
nn= nnsetup([size(X,2) 400 100 size(y,2)]);
nn.activation_function       = 'tanh_opt';
nn.learningRate              = 0.01;
nn.scaling_learningRate      = 1.0;
nn.inputZeroMaskedFraction   = 0.0;
nn.output                    = 'sigm';
opts.plot                    = 1;
opts.numepochs =   250;
% nn.W{1}=sae.ae{1}.W{1};
% % nn.W{2}=sae.ae{2}.W{1};
% nn.W{2}=nn1.W{1};
% nn.W{3}=nn1.W{2};
nn.weightPenaltyL2           = 0.01;            %  L2 regularization
nn=nntrain(nn,X,y,opts);

apt=zeros(size(Xtest,1),1);

%nn=nnff(nn,Xtest,apt);
Ytest=[zeros(100,1); ones(100,1)];
nn=nnff(nn,Xtest,Ytest);

fileID = fopen('Umiam.txt','w');



ar=nn.a{1,4};
for in=1:size(ar,1)
    if ar(in,1) >= 0.5
        ar(in,1) = 1;
        fprintf(fileID,'Photograph\r\n');
    else
        ar(in,1) = 0;
        fprintf(fileID,'Art\r\n');
    end
end

fclose(fileID);
err= sum(abs(ar-Ytest));
disp('Error');
disp(err);

% fileID = fopen('exp.txt','w');
% fprintf(fileID,'%6s %12s\n','x','exp(x)');

%Error= sum(abs(nn.e));

%disp('error on test set = '); disp(Error);
% a=min((mean(abs(nn.e))./mean([curYearY;ytest])).*100);
% if (a<mn) mn=a; end;
% e(i)= Error;
% figure
% plot(1:size(ytest,1),ytest(:,7),1:size(ytest,1),(ytest(:,7)-nn.e(:,7)));
% xlabel('time');
% ylabel('Cloud water content');
% legend('desired','predicted');
% title(['Predicted vs desired output for ' num2str(predictionYear) ' year '] ,'Fontsize',12);

% end
% disp('minimum errror = '); disp(mn);


