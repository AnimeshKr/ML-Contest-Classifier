function [nn, finalNN] = superVisedTrain_cwc_full_1layer( sae, U, initRan, initAvg, ran )
    
    no_features = size(sae.ae{1}.W{1},2) - 1;
    monthsOrfnights = 'fnights';
     
    
    if nargin == 5,
    
        %%%%%%%%%%%Reading input%%%%%%%%%%%%%%
        X = extractinput();
        X = X(1:end-1,:);
        y = csvread('cwcDaily.csv');
        y = y(5:end,:);
        assert(size(X,1) == size(y,1), 'dimensions not matching');
                        
        X = X - repmat( initAvg, size(X,1), 1 );
        X = X ./ repmat( initRan, size(X,1), 1 );
        X = X * U(:, (size(X,2) - no_features)+1:end );
        X = X ./ repmat( ran, size(X,1), 1 );


        if strcmp(monthsOrfnights,'months'),    %%%%%%%%%%%%Adding months data%%%%%%%%%%%
    
            months = csvread('monthsFrom1871.csv');
            months = months(5:end-1,:);
%             months = months(pos,:);
            assert(size(months,1) == size(y,1), 'dimensions of months are not matching');
            binMonths = repmat( months, 1, 12 ) == repmat( [1 2 3 4 5 6 7 8 9 10 11 12], size(months,1), 1 );
            save('traincwc_full.mat','X','y','binMonths','months');

        else %%%%%%%%%%%%Adding months data%%%%%%%%%%%

            fnights = csvread('fnightsFrom1871.csv');
            fnights = fnights(5:end-1,:);
%             fnights = fnights(pos,:);

            assert(size(fnights,1) == size(y,1), 'dimensions of fnights are not matching');
            binFnights = repmat( fnights, 1, 24 ) == repmat( [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24], size(fnights,1), 1 );
            save('traincwc_full.mat','X','y','binFnights','fnights');

        end;
        
%         save( 'traincwc_full.mat','X','y' );
    elseif nargin == 1,
        
        load traincwc_full.mat;

    end
    
    %%%%%%%%%Extracting data of required months or fnights
    if strcmp( monthsOrfnights, 'months' ),
        requiredMonths = [6,7,8,9];

        requiredPos = ismember( months, requiredMonths );
        X = X(requiredPos,:);
        y = y(requiredPos,:);
        binMonths = binMonths(requiredPos,:);
    else
        requiredFnights = [11,12,13,14,15,16,17,18];

        requiredPos = ismember( fnights, requiredFnights );
        X = X(requiredPos,:);
        y = y(requiredPos,:);
        binFnights = binFnights(requiredPos,:);
    end
     
    
    %%%%%%%%%%Running feedForward%%%%%%%%%
    sae.ae{1} = nnff(sae.ae{1}, X, X);
    features1 = sae.ae{1}.a{2};
    features1 = features1( : , 2:end );
   
    %%%%%%%%%Dividing into validation set%%%%
    trainSize = floor(0.91*size(X,1));
    
    features1Val = features1( trainSize+1:end, : );
    features1 = features1( 1:trainSize, : );
    
    binFnightsVal = binFnights( trainSize+1:end, : );
    binFnights = binFnights( 1:trainSize, : );
    
%     y = y * 100 ./ repmat( mean(y), size(y,1), 1 );
    y = 100*y(:,[5 80 140]);
    meanValues = mean(y)
    yval = y( trainSize+1:end, : );
    y = y( 1:trainSize, : );
    
    
    %%%%%%%%%Setting options%%%%%%%%%%%
    no_outputs = size(y,2);
    no_inputs = size(features1,2);
    hid = 150;
    nn = nnsetup( [ no_inputs hid no_outputs ] );
    
    nn.numepochs = 15;
    nn.learningRate{1} = 0.002;
    nn.learningRate{2} = 0.002;
    nn.output = 'linear';
    nn.activation_function = 'tanh_opt';
    nn.scaling_learningRate{1} = 1;
    nn.weightPenaltyL2 = 0;
    nn.momentum = 0.5;
    nn.scaling_momentum = 1;
    nn.weightPenaltyOnBias = 1;
    nn.adaptiveLearningRates = 'yes';
%     nn.opt_method = 'momentum';
   
    nn.W{2} = (rand(no_outputs, hid+size(binFnights,2)+1) - 0.5) * 10 * sqrt(6 / (hid + size(binFnights,2) + no_outputs));
    
    opts.plot = 1;
    opts.batchsize = floor(size(X,1)/80);
    nn.errorType = 'percent'; 
    nn.meanValues = meanValues;
    
    nn = nntrain( nn, features1, y, opts, features1Val, yval, binFnights, binFnightsVal ); 
    nn = nnff( nn, features1(1:4,:), y(1:4,:), binFnights(1:4,:) );
     
    y(1:4,:)
    nn.a{2}
    
    if strcmp( monthsOrfnights, 'months' ),
        finalNN = fineTune_single_1layer(nn, sae, X, [y; yval], binMonths);
    elseif strcmp( monthsOrfnights, 'fnights' )
        finalNN = fineTune_single_1layer(nn, sae, X, [y; yval], [binFnights;binFnightsVal]);
    else
        finalNN = fineTune_single_1layer(nn, sae, X, [y; yval]);
    end
        
end
