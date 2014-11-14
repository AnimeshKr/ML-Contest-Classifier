X1=read_script();
disp('read temp., pres');
disp(clock);
X1=double(X1);
X2=read_shumUV();
disp('read shum');
disp(clock);
X=[X1 X2];

[raindata ,A]=data_read();
A1=[]; ydata=[];c=0;
data=X(29220:end,:);
for i=1:54,
       A1=[A1;data(c+150:c+240,:)];
       ydata=[ydata;raindata(c+150:c+240,:)];
    if mod((i-2),4)==0,
        c=c+366;
    else
       c=c+365;
    end;
end;
ydata=ydata(:,1);
%--------tranforming the output ydata---------------------------------------

data=ydata;
gamma =skewness(data);
data=data+0.1;
data=log(data);
ndata=(data-mean(data))./std(data);
if gamma<0,
   y=min(ndata,-2/gamma);
elseif gamma==0
   disp('gamma=0') ;
else
   y=max(ndata,-2/gamma);
end;

fdata=(6/gamma).*(((gamma*y)/2 + 1).^1/3 - 1 )+ gamma/6;

%---------------------------------------------------------------------------
X=[A1 fdata];
rdata = [fdata(2:end,:);fdata(end,:)]; % next day rainfall


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
  
ndata=(U(:,1:k))'*X';
ndata=ndata';
disp('done PCA');
disp(clock);
% Normalizing:
for i  = 1:size(ndata,2),
     ndata(:,i)= (ndata(:,i)-mean(ndata(:,i)));
     ndata(:,i)= (ndata(:,i))/(max(ndata(:,i))-min(ndata(:,i)));
end;
xtest=ndata(end-100+1:end,:);
ndata=ndata(1:end-100,:);
ytest=rdata(end-100+1:end,:);
rdata=rdata(1:end-100,:);
%X_test=ndata(end-1000+1:end,:);
%ndata=ndata(1:end-1000,:);
for i=1:1
sae = saesetup([size(ndata,2) round(size(ndata,2)/4) round(size(ndata,2)/16)]);
sae.ae{1}.activation_function       = 'tanh_opt';
sae.ae{1}.learningRate              = 0.1;
sae.ae{1}.scaling_learningRate      = 1.0;
sae.ae{1}.inputZeroMaskedFraction   = 0.0;
sae.ae{1}.output                    = 'tanh_opt';

sae.ae{2}.activation_function       = 'tanh_opt';
sae.ae{2}.learningRate              = 0.1;
sae.ae{2}.scaling_learningRate      = 1.0;
sae.ae{2}.inputZeroMaskedFraction   = 0.0;
sae.ae{2}.output                    = 'tanh_opt';

% sae.ae{3}.activation_function       = 'tanh_opt';
% sae.ae{3}.learningRate              = 0.1;
% sae.ae{3}.scaling_learningRate      = 1.0;
% sae.ae{3}.inputZeroMaskedFraction   = 0.0;
% sae.ae{3}.output                    = 'tanh_opt';

opts.numepochs =   50;
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
end;


%-----------------------------------------
x_train=features2(:,2:end);

nn1=nnsetup([round(size(ndata,2)/16) 15 1]);
nn1.activation_function       = 'tanh_opt';
nn1.learningRate              = 0.001;
nn1.scaling_learningRate      = 1.0;
nn1.inputZeroMaskedFraction   = 0.0;
nn1.output                    = 'linear';
opts.plot                     =1;
nn1.W{1}=nn1.W{1};
opts.numepochs =   50;
nn1=nntrain(nn1,x_train,rdata,opts);

nn= nnsetup([size(ndata,2) round(size(ndata,2)/4) round(size(ndata,2)/16) 15 1]);
nn.activation_function       = 'tanh_opt';
nn.learningRate              = 0.001;
nn.scaling_learningRate      = 1.0;
nn.inputZeroMaskedFraction   = 0.0;
nn.output                    = 'linear';
opts.plot                    =1;
opts.numepochs =   50;
nn.weightPenaltyL2           = 0.1;            %  L2 regularization
nn.W{1}=sae.ae{1}.W{1};
nn.W{2}=sae.ae{2}.W{1};
%nn.W{3}=sae.ae{3}.W{1};
nn.W{3}=nn1.W{1}(:,1:end);
nn.W{4}=nn1.W{2};
nn=nntrain(nn,ndata,rdata,opts);


nn=nnff(nn,xtest,ytest);
a=nn.a{5};
yd=((((a-gamma/6).*(gamma/6)+1).^3)-1).*(2/gamma);
yd2=yd.*std(data)+mean(data);
y2=2.718.^yd2;
disp('error on test set = '); 
h=[ytest nn.a{5} nn.e];
disp(h);
disp(mean(mean(abs(nn.e))./mean(ytest)));
disp(clock);











