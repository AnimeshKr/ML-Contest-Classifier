function sae = saetrain(sae, x, opts)
  
    for i = 1 : numel(sae.ae);
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
%         if i==1,
%             sae.ae{i} = nntrain(sae.ae{i}, x, x, opts,valx,valy);
%         else
            sae.ae{i} = nntrain(sae.ae{i}, x, x, opts);
%         end;
        %disp('activation h '); disp([sae.ae{i}.a{2}]); pause;
        t = nnff(sae.ae{i}, x, x);  %after training we have obtained weights and now using those to find out the activations.
        x = t.a{2};
        %remove bias term
        x = x(:,2:end);
    end
end
