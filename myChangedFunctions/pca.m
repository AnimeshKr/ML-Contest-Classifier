function [ X , U , error , D] = pca( input , var )
    
    [U,D] = eig( input' * input / size(input,1) );
    
    %Calculating variance and hence number of units
    D = diag(D);
    totalSum = sum( D );
    curSum = 0;
%     for i = 1:size(input,2)
%         D(i,i)
%     end
    for i = size(input,2):-1:1,
        curSum = curSum + D(i);
%         curSum/totalSum
        if ( curSum / totalSum > var ),
            break;
        end
    end
    
    X = input * U( : , i:end );
    initX = X * U(:,i:end)';
    error = sum(sum(abs(initX - input))) * 100 / sum(abs(input(:)));

end