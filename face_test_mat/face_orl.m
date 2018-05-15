% ***
% ******************************************************************************
% * @filename:  face recognition based on PCA and Euclidean distance on ORL
% * @author  : tjwx
% * @version :
% * @date    : 2018.04.06
% * @brief   : This file provides all the  **** functions.
% * @reference:   https://zhuanlan.zhihu.com/p/26652435
% ******************************************************************************
% ***
clear all
tic;

%���������ռ�����
Img_Mat = [ ];
Train_Mat = [ ];
Train_Labels = [ ];
Test_Mat = [ ];
Test_Labels = [ ];
for i = 1:40    
    for j = 1:10
    str = strcat('I:\ѧϰ����\ģʽʶ��\6ϵ\��ҵ2\�ڶ��δ���ҵ���\ORL������\ORL92112\bmp\s',  ...
                         int2str(i),'\',int2str(j),'.bmp');
    temp_mat = imread(str);
    [r,c] = size(temp_mat);
    temp_mat = reshape(temp_mat,r*c,1);   %��ͼƬת��Ϊһ��������������ÿ����һ��ά��
    if j == 4|| j ==8
        Test_Mat = [Test_Mat, temp_mat];
        Test_Labels = [Test_Labels; i,j];
    else
        Train_Mat = [Train_Mat, temp_mat];
        Train_Labels = [Train_Labels; i,j];
    end
    Img_Mat = [Img_Mat,temp_mat];
    end
end

%����ѵ�����������������ռ�,ע����ΪA������ԶԶ��������������
%�˴���A'A������ֵ��AA'������任�����̼���ʱ��
differ_mat = bsxfun(@minus, double(Train_Mat), mean(Train_Mat,2));
L_Mat = (differ_mat' * differ_mat);
[eiv, eic] = eig(L_Mat);   %��ȡ��������eiv�Լ�����ֵeic ,����ֵ����

% ������ֵѡȡ��k�����ɷ�,����ȡ�������ռ��Լ�ԭ������ͶӰ�ռ������ֵ
SelectThrehold = 0.95;
Select_sum = 0;
diag_eic = diag(eic);
Sum_total = sum(diag_eic);
L_eig_vec = [ ];
for i = size(diag_eic):-1:1
     Select_sum = Select_sum +diag_eic(i,1);
    if (Select_sum / Sum_total > SelectThrehold)
      L_eig_vec =eiv(:,i:size(diag_eic));    %ѡȡ����ֵ���״�����ֵ������������
      break;
    end
end
for i = 1:size(L_eig_vec,2)      %����������λ��  
    L_eig_vec(:,i) = L_eig_vec(:,i) / norm(L_eig_vec(:,i));  
end  
Ei_Face = differ_mat * L_eig_vec;     %�õ�Э������������������ɵ��������ռ�
Train_Project = Ei_Face' * differ_mat;
test_temp =  bsxfun(@minus, double(Test_Mat), mean(Test_Mat,2));
Test_Project = Ei_Face' * test_temp;

%����ŷ�Ͼ���,���ʶ��
index = 0;
match_index =[];
for j =1: size(Test_Project,2)
    com_dist = [ ];
    for i = 1:size(Train_Project,2)
        vec_dist = norm(Test_Project(:,j) - Train_Project(:,i));
        com_dist = [com_dist, vec_dist];
    end
    [~,match_index] = min(com_dist);
    if Train_Labels(match_index,1) == Test_Labels(j,1)
        index = index + 1;
    end
end
acc = index / size(Test_Project,2) ;
toc;
temp = sprintf('The Accuracy by Euclidean distance and PCA is   %f ',acc);
disp(temp);





