clc;
close all;
clear all;
no_files=200;

Xtest=zeros(96*96*3,no_files);
for name = 1: no_files
   
       filename=sprintf('1 (%d).jpg', name);
       frame=imread(filename);
       frame = imresize(frame,[96 96]);
       frame_size = size(frame);
       if numel(frame_size)==2
          Xtest(:,name) = Xtest(:,name-1) ;
       else
            allpixels = reshape(frame, frame_size(1)*frame_size(2), frame_size(3));
            allpixels=allpixels';
            allpixels=reshape(allpixels,[],1);
            Xtest(:,name)=allpixels;
       end
       
end
  % art=0 photo=1;
save('testFile.mat','Xtest');