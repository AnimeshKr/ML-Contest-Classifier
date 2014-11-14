function [finalNN] = fineTune_single_1layer( nn, sae, X, y, extraInputs )
    
%     pos = randperm( size(X, 1) );
%     X = X(pos,:);   y = y(pos,:);

    trainSize = ceil( 0.91 * size(X,1) );
    Xval = X( trainSize+1:end, : );
    yval = y( trainSize+1:end, : );
    
    X = X( 1:trainSize, : );
    y = y( 1:trainSize, : );
    
    if nargin == 5,
%         extraInputs = extraInputs(pos,:);
        extraInputsVal = extraInputs( trainSize+1:end, : );
        extraInputs = extraInputs( 1:trainSize, : );
    end
    
    no_inputs = size(sae.ae{1}.W{1},2)-1;
    Deephid1 = size(sae.ae{1}.W{1},1);
    hid = size(nn.W{1},1);
    no_outputs = size(nn.W{2},1);
    
    finalNN = nnsetup( [ no_inputs Deephid1 hid no_outputs ] );
%     finalNN.weightPenaltyL2 = 0.01;
    finalNN.output = 'linear';
    finalNN.activation_function = 'tanh_opt';
    
    finalNN.W{1} = sae.ae{1}.W{1};
    finalNN.W{2} = nn.W{1};
    finalNN.W{3} = nn.W{2};
    
    finalNN.numepochs = 25;
    options.batchsize = floor(size(X,1)/70); 
    options.plot = 1;
    
    finalNN.errorType = 'percent';
    finalNN.meanValues = mean( [y; yval] );
 
%     finalNN.weightPenaltyL1 = 0.001;
%     finalNN.regularizationType = 'L1';
    
%     extraInputsVal = extraInputs(1:1000,:);
%     Xval = X(1:1000,:);
%     yval = y(1:1000,:);
    
    if nargin == 5,
        finalNN = nntrain( finalNN, X, y, options, Xval, yval, extraInputs, extraInputsVal );
        finalNN = nnff( finalNN, X(1:150,:), y(1:150,:), extraInputs(1:150,:) );
        y(1:10,:)
        finalNN.a{4}
        finalNN = nnff( finalNN, Xval(1:150,:), yval(1:150,:), extraInputsVal(1:150,:) );
        yval(1:10,:)
        finalNN.a{4}
        plot(1:size(finalNN.a{4},1), finalNN.a{4}(1:150,1), 1:size(finalNN.a{4},1), yval(1:150,1));
    else
        finalNN = nntrain( finalNN, X, y, options, Xval, yval );
        finalNN = nnff( finalNN, X(1:10,:), y(1:10,:) );
        y(1:10,:)
        finalNN.a{4}
        finalNN = nnff( finalNN, Xval(1:10,:), yval(1:10,:) );
        yval(1:10, :)
        finalNN.a{4}
    end
    
end
