% ***
% ******************************************************************************
% * @fun_name :特征数据规范化 
% * @author   : tjwx
% * @arg_in   : faceMat: 需要进行规范化的图像数据,按行规范
% *                   lowvec:  原来的最小值  
% *                   upvec:   原来的最大值  
% * @arg_out  : 
% * @ATTENTION:
% ******************************************************************************
% ***
function [ scaledface] = scaling( faceMat,lowvec,upvec )  
%特征数据规范化  
upnew=1;  
lownew=-1;  
[m,n]=size(faceMat);  
scaledface=zeros(m,n);  
for i=1:m  
    scaledface(i,:)=lownew+(faceMat(i,:)-lowvec)./(upvec-lowvec)*(upnew-lownew);  
end  
end  


