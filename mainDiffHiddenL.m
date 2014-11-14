X1=read_script();
disp('read temp., pres, cldwtr');
disp(clock);
X1=double(X1);
X2=read_shumUV();
disp('read shum, u, v');
disp(clock);

X=[X1 X2];
%ndata=X;
%------------------------
s = std(X,1,1);
  for i  = 1:size(X,2),
    X(:,i)= (X(:,i)-mean(X(:,i)))/s(1,i); 
  end; 
  
  Sig=cov(X);
  [U,S,V]=svd(Sig);
  k=1;
  %disp('size of S');disp(size(S));
  while (sum(diag(S(1:k,1:k))))/sum(diag(S)) < 0.999;
      k=k+1;
      %disp('k= ');disp(k);
  end;
  %disp('learning rate should be:');
  %disp(1/max(diag(S)));
  %pause;
ndata=(U(:,1:k))'*X';
ndata=ndata';
disp('done PCA');
disp(clock);
% Normalizing:
for i  = 1:size(ndata,2),
     ndata(:,i)= (ndata(:,i)-mean(ndata(:,i)));
     ndata(:,i)= (ndata(:,i))/(max(ndata(:,i))-min(ndata(:,i)));
end;
X_test=ndata(end-1000+1:end,:);
ndata=ndata(1:end-1000,:);

% for i  = 1:size(X,2),
%      X(:,i)= (X(:,i)-mean(X(:,i)))/(max(X(:,i))-min(X(:,i)));
%      X(:,i)= (X(:,i)-min(X(:,i)));
%      X(:,i)= (X(:,i))/(max(X(:,i))-min(X(:,i)));
% end;
% ndata=X;
% Training the autoencoder starting:

sae = saesetup([size(ndata,2) round(size(ndata,2)/4) round(size(ndata,2)/16)]);
sae.ae{1}.activation_function       = 'tanh_opt';
sae.ae{1}.learningRate              = 0.1;
sae.ae{1}.scaling_learningRate      = 1.0;
sae.ae{1}.inputZeroMaskedFraction   = 0.0;
sae.ae{1}.output                    = 'tanh_opt';
sae.ae{1}.weightPenaltyL2           = 0.0;            %  L2 regularization

sae.ae{2}.activation_function       = 'tanh_opt';
sae.ae{2}.learningRate              = 0.05;
sae.ae{2}.scaling_learningRate      = 0.9;
sae.ae{2}.inputZeroMaskedFraction   = 0.0;
sae.ae{2}.output                    = 'tanh_opt';
sae.ae{2}.weightPenaltyL2           = 0.0;            %  L2 regularization
% 
% sae.ae{3}.activation_function       = 'tanh_opt';
% sae.ae{3}.learningRate              = 0.005;
% sae.ae{3}.scaling_learningRate      = 1.0;
% sae.ae{3}.inputZeroMaskedFraction   = 0.0;
% sae.ae{3}.output                    = 'tanh_opt';
% nn.weightPenaltyL2           = 0.0;            %  L2 regularization

opts.numepochs =   30;
opts.batchsize = 100;
opts.plot = 1;
sae = saetrain(sae,ndata , opts);
visualize(sae.ae{1}.W{1}(:,2:end)');

sae.ae{1}=nnff(sae.ae{1},ndata,ndata);
features1= sae.ae{1}.a{2};

sae.ae{2}=nnff(sae.ae{2},features1(:,2:end),features1(:,2:end));
features2= sae.ae{2}.a{2};

% sae.ae{3}=nnff(sae.ae{3},features2(:,2:end),features2(:,2:end));
% features3= sae.ae{3}.a{2};

%collecting rain data
[raindata,A]=data_read();
raindata = [raindata(2:end,:);raindata(end,:)]; %  because we are predicting next day rainfall
Y_test=raindata(end-1000+1:end,:);
raindata=raindata(1:end-1000,:);
x_train=[features2(29220:end,2:end)];

%processing for monsoon months data:
c=0;

% for i=1:54,
%     x_1=[x_1;x_train(c+140:c+245,:)];
%     r_1=[r_1;raindata(c+140:c+245,:)];
%     c=c+365;   
% end;  
nn1=nnsetup([round(size(ndata,2)/16) 7]);
nn1.activation_function       = 'tanh_opt';
nn1.learningRate              = 0.001;
nn1.scaling_learningRate      = 1.0;
nn1.inputZeroMaskedFraction   = 0.0;
nn1.output                    = 'linear';
opts.plot                     =1;
opts.numepochs =   50;
nn1.W{1}=nn1.W{1};
nn1=nntrain(nn1,x_train,raindata,opts);

% Final Training:
nn= nnsetup([size(ndata,2) round(size(ndata,2)/4)  round(size(ndata,2)/16) 7]);
nn.activation_function       = 'tanh_opt';
nn.learningRate              = 0.001;
nn.scaling_learningRate      = 1.0;
nn.inputZeroMaskedFraction   = 0.0;
nn.weightPenaltyL2           = 0.1;            %  L2 regularization
nn.output                    = 'linear';
opts.plot                     =1;
opts.numepochs =   50;
nn.W{1}=sae.ae{1}.W{1};
nn.W{2}=sae.ae{2}.W{1};
%nn.W{3}=sae.ae{3}.W{1};
nn.W{3}=nn1.W{1};
nn=nntrain(nn,ndata(29220:end,:),raindata,opts);

nn=nnff(nn,X_test,Y_test);
disp('error on test set = '); disp(nn.e);

disp(clock);



