% ***
% ******************************************************************************
% * @filename:  face recognition based on PCA and Euclidean distance on PIE
% * @author  : tjwx
% * @version :
% * @date    : 2018.05.14
% * @brief   : This file provides all the  **** functions.
% * @reference:   https://zhuanlan.zhihu.com/p/26652435
% ******************************************************************************
% ***
clear all
tic;
%���������ռ�����
Fea_Mat = [ ];
Gnd_Mat = [ ];
Istest_Mat = [ ];
Train_Mat = [ ];
Train_Labels = [ ];
Test_Mat = [ ];
Test_Labels = [ ];
for i = 1:5    
    str = strcat(' ',int2str(i),'_64x64.mat');
    load(str);
    Fea_Mat = [Fea_Mat ; fea];
    Gnd_Mat = [Gnd_Mat ; gnd];
    Istest_Mat = [Istest_Mat ; isTest ]; 
end
% �����Ż�
for i = 1 : length(Istest_Mat)
   if Istest_Mat(i , 1) == 1
        Test_Mat = [Test_Mat;Fea_Mat(i , :)];
        Test_Labels = [Test_Labels; double(Gnd_Mat(i , :))];
   else
        Train_Mat = [Train_Mat;Fea_Mat(i , :)];
        Train_Labels = [Train_Labels; double(Gnd_Mat(i , :))];
   end
end
Train_Mat = Train_Mat' ;
Test_Mat = Test_Mat';
%����ѵ�����������������ռ�,ע����ΪA������ԶԶС������������
%�˴�����A'A������ֵ��AA'������任
differ_mat = bsxfun(@minus, double(Train_Mat), mean(Train_Mat,2));
L_Mat = (differ_mat* differ_mat');
[eiv, eic] = eig(L_Mat);   %��ȡ��������eiv�Լ�����ֵeic ,����ֵ����

% ������ֵѡȡ��k�����ɷ�,����ȡ�������ռ��Լ�ԭ������ͶӰ�ռ������ֵ
SelectThreshold = 0.95;
Select_sum = 0;
diag_eic = diag(eic);
Sum_total = sum(diag_eic);
L_eig_vec = [ ];
for i = size(diag_eic):-1:1
     Select_sum = Select_sum +diag_eic(i,1);
    if (Select_sum / Sum_total > SelectThreshold)
      L_eig_vec =eiv(:,i:size(diag_eic));    %ѡȡ����ֵ���״�����ֵ������������
      break;
    end
end
for i = 1:size(L_eig_vec,2)      %����������λ��  
    L_eig_vec(:,i) = L_eig_vec(:,i) / norm(L_eig_vec(:,i));  
end  
Ei_Face =L_eig_vec;     %�õ�Э������������������ɵ��������ռ�
Train_Project = Ei_Face' * differ_mat;
test_temp =  bsxfun(@minus, double(Test_Mat), mean(Test_Mat,2));
Test_Project = Ei_Face' * test_temp;

%����ŷ�Ͼ���,���ʶ��
index = 0;
match_index =zeros(size(Test_Project,2),1);
for j =1: size(Test_Project,2)
    com_dist = [ ];
    for i = 1:size(Train_Project,2)
        vec_dist = norm(Test_Project(:,j) - Train_Project(:,i));
        com_dist = [com_dist, vec_dist];
    end
    [~ , match_index(j)] = min(com_dist);
    if Train_Labels(match_index(j),1) == Test_Labels(j,1)
        index = index + 1;
    end
end
acc = index / size(Test_Project,2) ;

%����libSVM ��ʵ�ַ���
lowvec_train = min(Train_Project);  
upvec_train = max(Train_Project);  
Train_svm = scaling(Train_Project , lowvec_train , upvec_train);
lowvec_test = min(Test_Project);  
upvec_test = max(Test_Project);  
Test_svm = scaling(Test_Project , lowvec_test , upvec_test);
[~,bestc,bestg] = SVMcg(Train_Labels,Train_svm' , -8, 4, -8 , 8 , 5, 1, 1, 4.5);  %ת��Ϊ��ά��
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model = svmtrain(Train_Labels , Train_svm' , cmd);
[plabel ,  accuracy , dec] = svmpredict(Test_Labels , Test_svm' , model);
toc ;
temp1 = sprintf('The Accuracy by Euclidean distance and PCA is   %f ',acc);
disp(temp1);
temp = sprintf('The Accuracy by SVM and PCA is  %f ',accuracy(1,1));
disp(temp);
save  result_pie.mat  acc  Ei_Face  accuracy  model



