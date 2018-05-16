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

%读入样本空间数据
Img_Mat = [ ];
Train_Mat = [ ];
Train_Labels = [ ];
Test_Mat = [ ];
Test_Labels = [ ];
for i = 1:40    
    for j = 1:10
    str = strcat('I:\学习事务\模式识别\6系\作业2\第二次大作业相关\ORL人脸库\ORL92112\bmp\s',  ...
                         int2str(i),'\',int2str(j),'.bmp');
    temp_mat = imread(str);
    [r,c] = size(temp_mat);
    temp_mat = reshape(temp_mat,r*c,1);   %将图片转化为一个列向量，这样每行是一个维度
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

%利用训练样本构建特征脸空间,注意因为A的行数远远大于列数，所以
%此处用A'A的特征值与AA'相等做变换，缩短计算时间
differ_mat = bsxfun(@minus, double(Train_Mat), mean(Train_Mat,2));
L_Mat = (differ_mat' * differ_mat);
[eiv, eic] = eig(L_Mat);   %求取特征向量eiv以及特征值eic ,特征值升序

% 按照阈值选取第k个主成分,并求取特征脸空间以及原数据在投影空间的坐标值
SelectThrehold = 0.95;
Select_sum = 0;
diag_eic = diag(eic);
Sum_total = sum(diag_eic);
L_eig_vec = [ ];
for i = size(diag_eic):-1:1
     Select_sum = Select_sum +diag_eic(i,1);
    if (Select_sum / Sum_total > SelectThrehold)
      L_eig_vec =eiv(:,i:size(diag_eic));    %选取特征值贡献大于阈值的特征向量组
      break;
    end
end
for i = 1:size(L_eig_vec,2)      %特征向量单位化  
    L_eig_vec(:,i) = L_eig_vec(:,i) / norm(L_eig_vec(:,i));  
end  
Ei_Face = differ_mat * L_eig_vec;     %得到协方差矩阵的特征向量组成的特征脸空间
Train_Project = Ei_Face' * differ_mat;
test_temp =  bsxfun(@minus, double(Test_Mat), mean(Test_Mat,2));
Test_Project = Ei_Face' * test_temp;

%计算欧氏距离,完成识别
index = 0;
match_index =[];
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


%调用libSVM ，实现分类
lowvec_train = min(Train_Project);  
upvec_train = max(Train_Project);  
Train_svm = scaling(Train_Project , lowvec_train , upvec_train);
lowvec_test = min(Test_Project);  
upvec_test = max(Test_Project);  
Test_svm = scaling(Test_Project , lowvec_test , upvec_test);
[~,bestc,bestg] = SVMcgForClass(Train_Labels,Train_svm' , -8, 4, -8 , 8 , 5, 1, 1, 4.5);  %转化为列维度
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model = svmtrain(Train_Labels , Train_svm' , cmd);
[plabel ,  accuracy , dec] = svmpredict(Test_Labels , Test_svm' , model);
toc ;
temp1 = sprintf('The Accuracy by Euclidean distance and PCA is   %f ',acc);
disp(temp1);
temp = sprintf('The Accuracy by SVM and PCA is  %f ',accuracy(1,1));
disp(temp);
save model_orl.mat  model


