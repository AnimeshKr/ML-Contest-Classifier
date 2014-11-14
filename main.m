datamatrix=xlsread('tempNpres.csv');
%  tempdata= (datamatrix(:,1:289))./500;
%  presdata= (datamatrix(:,290:578))./200000;

 tempdata= (datamatrix(4176:(4176+2816),1:289));
 presdata= (datamatrix(4176:(4176+2816),290:578));
 %tdata= tempdata(:,1:10);
 X= [tempdata(:,:) presdata(:,:)];
 %X_test=[tempdata(2718:end,:) presdata(2718:end,:)];
 X=double(X);
 %X_test=double(X_test);
 %X= zeros(20000,2);
 %X(:,i)= (X(:,i)-mean(X(:,i)))./(5*std(X(:,i)));
 %X=[reshape(tdata,[],1) reshape(pdata,[],1)];
 %X(:,i)= (X(:,i)-mean(X(:,i)))./(max(X(:,i))-min(X(:,i)));
 
%  for i  = 1:578,
%     X(:,i)= (X(:,i)-mean(X(:,i)))./(max(X(:,i))-min(X(:,i))); 
%     X(:,i)= (X(:,i)-min(X(:,i)));
%     X(:,i)= (X(:,i))/(max(X(:,i))-min(X(:,i)));
%  end;
%   for i  = 1:578,
%     X_test(:,i)= (X_test(:,i)-mean(X_test(:,i)))./(max(X_test(:,i))-min(X_test(:,i))); 
%     X_test(:,i)= (X_test(:,i)-min(X_test(:,i)));
%     X_test(:,i)= (X_test(:,i))/(max(X_test(:,i))-min(X_test(:,i)));
%  end;
 
 s = std(X,1,1);
  for i  = 1:578,
    X(:,i)= (X(:,i)-mean(X(:,i)))/s(1,i); 
  end; 
  
  Sig=cov(X);
  [U,S,V]=svd(Sig);
  k=1;
  %disp('size of S');disp(size(S));
  while (sum(diag(S(1:k,1:k))))/sum(diag(S)) < 0.9999;
      k=k+1;
      %disp('k= ');disp(k);
  end;
  %pause;
ndata=(U(:,1:k))'*X';
ndata=ndata';
%disp([ndata(:)]);

for i  = 1:size(ndata,2),
     ndata(:,i)= (ndata(:,i)-min(ndata(:,i)));
     ndata(:,i)= (ndata(:,i))/(max(ndata(:,i))-min(ndata(:,i)));
end;

X_test= ndata(2718:end,:);
ndata= ndata(1:2717,:);

sae = saesetup([size(ndata,2) round(size(ndata,2)/2) round(size(ndata,2)/4) round(size(ndata,2)/8)]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 0.05;
sae.ae{2}.scaling_learningRate      = 1.0;
sae.ae{1}.inputZeroMaskedFraction   = 0.0;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 0.05;
sae.ae{2}.scaling_learningRate      = 1.0;
sae.ae{2}.inputZeroMaskedFraction   = 0.0;

sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 0.05;
sae.ae{3}.scaling_learningRate      = 1.0;
sae.ae{3}.inputZeroMaskedFraction   = 0.0;

opts.numepochs =   60;
opts.batchsize = 100;
opts.plot = 1;
sae = saetrain(sae,ndata , opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

sae.ae{1}=nnff(sae.ae{1},ndata,ndata);
features1= sae.ae{1}.a{2};

sae.ae{2}=nnff(sae.ae{2},features1(:,2:end),features1(:,2:end));
features2= sae.ae{2}.a{2};

sae.ae{3}=nnff(sae.ae{3},features2(:,2:end),features2(:,2:end));
features3= sae.ae{3}.a{2};

% Use the SDAE to initialize a FFNN

raindata=data_read();
% s2 = std(raindata,1,1);
% raindata(:,1)= (raindata(:,1)-mean(raindata(:,1)))./s2;
rdata=raindata;
raindata(:,1)= (raindata(:,1)-mean(raindata(:,1)))./(max(raindata(:,1))-min(raindata(:,1))); 
raindata(:,1)= (raindata(:,1)-min(raindata(:,1)));
raindata(:,1)= (raindata(:,1))/(max(raindata(:,1))-min(raindata(:,1)));
y_test=raindata(2718:2817,1);
raindata=raindata(1:2717,1);
%disp('raindata');disp(raindata(:)); pause;

% last layer(linear)
rdata=rdata(1:2717,1);

disp('last linear training');pause;
nn1=nnsetup([round(size(ndata,2)/8) 1]);
opts.plot = 1;
nn1.activation_function='sigm';
nn1.learningRate                     = 0.05;   
nn1.output= 'linear';
disp('weights of the linear learning for nn1 last layer.');disp(nn1.W{1});pause;

nn1=nntrain(nn1,features3(:,2:end),rdata,opts);


% Final ANN
nn = nnsetup([size(ndata,2) round(size(ndata,2)/2) round(size(ndata,2)/4) round(size(ndata,2)/8) 1]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.05;
nn.scaling_learningRate      = 1.0;
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.W{4}= nn1.W{1};
opts.numepochs =   50;
opts.batchsize = 70;
%Train the FFNN

disp('weights of the linear learning for nn1 last layer.');disp(nn.W{3});pause;
nn.output='linear';

nn = nntrain(nn, ndata, rdata, opts);
% disp('weight 1');disp(nn.W{1}); pause;
%disp('activation 3');disp(nn.a{3}); pause;
%disp('weight 3');disp(nn.W{3}); pause;
%disp('activation 4');disp(nn.a{4}); pause;
% disp('activation 2');disp(nn.a{2}); pause;

%disp('y_test=');disp(y_test);
%disp('activation 4');disp(nn.a{4}); pause;
%[er, bad] = nntest(nn, X_test, y_test);

%assert(er < 0.16, 'Too big error');
%disp(er);