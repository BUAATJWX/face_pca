% ***
% ******************************************************************************
% * @fun_name :�������ݹ淶�� 
% * @author   : tjwx
% * @arg_in   : faceMat: ��Ҫ���й淶����ͼ������,���й淶
% *                   lowvec:  ԭ������Сֵ  
% *                   upvec:   ԭ�������ֵ  
% * @arg_out  : 
% * @ATTENTION:
% ******************************************************************************
% ***
function [ scaledface] = scaling( faceMat,lowvec,upvec )  
%�������ݹ淶��  
upnew=1;  
lownew=-1;  
[m,n]=size(faceMat);  
scaledface=zeros(m,n);  
for i=1:m  
    scaledface(i,:)=lownew+(faceMat(i,:)-lowvec)./(upvec-lowvec)*(upnew-lownew);  
end  
end  


