clear all;
clc;

[X,y]=data_extract;

% PCA Starting
initAvg = mean(X);
X = X - repmat(initAvg, size(X,1), 1);
initRan = sqrt( mean(X.^2) );
X = X ./ repmat(initRan, size(X,1), 1);
%[X, U, error, ~] = pca(X,0.9999);

% Normalizing:
ran1 = 3.25*sqrt( var(X) );
X = X ./ repmat(ran1, size(X,1), 1);

startYear = 2007;
endYear = 2012;
prevYears = 60;

input_layer_size = size(X,2);
hidden_layer_size = 400;
num_labels = size(X,2);
lambda= 0.05;
n=0;e=0;
y=y*100;
origX=X;
origY=y;
mn=999;

for i = 1:(endYear-startYear)+1,

    predictionYear = startYear + i - 1;
    errors(i,1) = predictionYear;
    prevErrors(i,1) = predictionYear;

    Xtest = origX(end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , :);
    ytest = origY(end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , :);
    
%     curYearX = origX( end - ( ( (2011+prevYears)-predictionYear )*122+121 ) : end - ( (2011-predictionYear)*122+122 ), : );
%     curYearY = origY( end - ( ( (2011+prevYears)-predictionYear )*122+121 ) : end - ( (2011-predictionYear)*122+122 ), : );
     
    curYearX = origX( 1 : end - ( (2012-predictionYear)*122+122 ), : );
    curYearY = origY( 1 : end - ( (2012-predictionYear)*122+122 ), : );
        initAvg = mean(curYearX);
    curYearX = curYearX - repmat(initAvg, size(curYearX,1), 1);
    Xtest = Xtest - repmat(initAvg, size(Xtest,1), 1);
    initRan = std(curYearX);
    curYearX = curYearX ./ repmat(initRan, size(curYearX,1), 1);
    Xtest = Xtest ./ repmat(initRan, size(Xtest,1), 1);


opts.numepochs =   50;
opts.batchsize = 100;
opts.plot = 1;

% Final Training:
nn= nnsetup([size(curYearX,2),400, 200 ,size(curYearY,2)]);
nn.activation_function       = 'tanh_opt';
nn.learningRate              = 0.001;
nn.scaling_learningRate      = 1.0;
nn.inputZeroMaskedFraction   = 0.0;
nn.output                    = 'linear';
opts.plot                    = 1;
opts.numepochs =   50;
nn.weightPenaltyL2           = 0.1;            %  L2 regularization
nn=nntrain(nn,curYearX,curYearY,opts);

nn=nnff(nn,Xtest,ytest);
Error= mean(mean(abs(nn.e))./mean([curYearY;ytest]));
disp('error on test set = '); disp(Error);
e(i)= Error;
a=min((mean(abs(nn.e))./mean([curYearY;ytest])).*100);
if (a<mn) mn=a; end;
figure
plot(1:size(ytest,1),ytest(:,7),1:size(ytest,1),(ytest(:,7)-nn.e(:,7)));
xlabel('time');
ylabel('Cloud water content');
legend('desired','predicted');
title(['Predicted vs desired output for ' num2str(predictionYear) ' year '] ,'Fontsize',12);

end
disp('minimum errror = '); disp(mn);


