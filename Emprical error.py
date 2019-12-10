

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:26:49 2019

@author: arifr
"""

import math
import numpy as np
import os
from PIL import Image
from collections import defaultdict






def int_img(img_arr):


    ii = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1))
    row_sum = np.zeros(img_arr.shape)
    for x in range(img_arr.shape[1]):
        for y in range(img_arr.shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
            ii[y+1, x+1] = ii[y+1, x-1+1] + row_sum[y, x]
    return ii




def rect_area(ii,x,y,w,h):
    if w==0 and h==0:
        return ii(x,y)
    a=(x,y)
    b=(x+w,y)
    c=(x,y+h)
    d=(x+w,y+h)

    return ii[d]-ii[b] -ii[c]+ ii[a]





def f1_val(img,x,y,w,h):
    return rect_area(img,x,y,w,h/2)- rect_area(img,x,y+(h/2),w,h/2)

def f2_val(img,x,y,w,h):
    return +rect_area(img,x,y,w/2,h)-rect_area(img,x+(w/2),y,w/2,h)


def f3_val(img,x,y,w,h):
    return rect_area(img,x,y,w/3,h) -rect_area(img,x+(w/3),y,w/3,h) +rect_area(img,x+(2*w/3),y,w/3,h)

def f4_val(img,x,y,w,h):
    return rect_area(img,x,y,w,h/3)- rect_area(img,x,y+(h/3),w,h/3) + rect_area(img,x,y+(2*h/3),w,h/3)


def f5_val(img,x,y,w,h):
    return rect_area(img,x,y,w/2,h/2)+rect_area(img,x+(w/2),y+(h/2),w/2,h/2) -rect_area(img,x,y+(h/2),w/2,h/2)-rect_area(img,x+(w/2),y,w/2,h/2) 







    

def features1_creation():
    features1 = [] 
    count=0
    w_step, h_step = 1,2
    for w in range(1, 20, w_step):
        for h in range(2, 20, h_step):
            for pos_x in range(20 - w):
                for pos_y in range(20 - h):
                    features1.append((1,pos_x, pos_y, w, h)) 
    
    return features1


def features2_creation():
    features2 = [] 
    count=0
    w_step, h_step = 2,1
    for w in range(2, 20, w_step):
        for h in range(1, 20, h_step):
            for pos_x in range(20 - w):
                for pos_y in range(20 - h):
                    features2.append((2,pos_x, pos_y, w, h)) 
    
    return features2


def features3_creation():
    features3 = [] 
    count=0
    w_step, h_step = 3,1
    for w in range(3, 20, w_step):
        for h in range(1, 20, h_step):
            for pos_x in range(20 - w):
                for pos_y in range(20 - h):
                    features3.append((3,pos_x, pos_y, w, h)) 
    
    return features3


def features4_creation():
    features4 = [] 
    count=0
    w_step, h_step = 1,3
    for w in range(1, 20, w_step):
        for h in range(3, 20, h_step):
            for pos_x in range(20 - w):
                for pos_y in range(20 - h):
                    features4.append((4,pos_x, pos_y, w, h)) 
    
    return features4


def features5_creation():
    features5 = [] 
    count=0
    w_step, h_step = 2,2
    for w in range(2, 20, w_step):
        for h in range(2, 20, h_step):
            for pos_x in range(20 - w):
                for pos_y in range(20 - h):
                    features5.append((5,pos_x, pos_y, w, h)) 
    
    return features5

def im_arr(l):
    arr_im = []
    for i in l :
            arr = np.array(i, dtype=np.float64)
            arr /= 255.
            arr_im.append(arr)
    return arr_im



def train(ft_arr,nft_arr,num_of_classifers):

    
    global labels
    global fea_img
    global train_res
    global wts                              #new
    global features  

    full_train= ft_arr+nft_arr

    added_set=set()
    res=[]
    ans=[]

    seen_ee= set()
    seen_fp= set()
    seen_fn= set()
    alphas=[]

    err_allf=[[0]*len(full_train) for i in range(len(features))]
    c1=0
    c2=0
    for k in range(num_of_classifers):

        print "currently running adaboost round " + str(k+1) 

        errors= [0]*len(features)
        tot = sum(wts)
        wts=[i*(1./tot) for i in wts]

        train_all(fea_img,wts)


        best_clf_ee, best_error_ee, best_accuracy_ee = None, float('inf'), None

        for i in range(len(features)):

            error, accuracy = 0, []
            acc=0
            false_positive=0
            false_negative=0
            for j in range(len(full_train)):
                prediction=1 if fea_img[i][j]*ftp[features[i]][1]  < ftp[features[i]][0]*ftp[features[i]][1] else -1
                if labels[j]==-1 and prediction==1:
                    false_positive+=1
  
                    
                if labels[j]==1 and prediction==-1:
                    false_negative+=1


                  
                correctness=abs(prediction-labels[j])
                accuracy.append(correctness)
                err_allf[i][j]=wts[j]*correctness
                if prediction==labels[j]:
                    acc+=1

            error_ee= sum(err_allf[i]) if features[i] not in seen_ee else float('inf')
            if error_ee==0:
                c1+=1
            else:
                c2+=1
                

            
            if error_ee<best_error_ee:
                best_clf_ee, best_error_ee, best_accuracy_ee,index_ee,accf_ee,fn_ee,fp_ee = features[i], error_ee, accuracy,i,float(acc)/len(full_train),false_negative,false_positive

        print "Best classifer in this round is ", best_clf_ee
        print "accuracy of the chosen feature is ", accf_ee
        x=(best_clf_ee,accf_ee,index_ee,fp_ee,fn_ee)

        
        seen_ee.add(best_clf_ee)
        beta=2
        e=best_error_ee
        
        
        beta= float(e)/(1- float(e))


    
        for i,wt in enumerate(wts):
            if accuracy[i]==0:
                wts[i]= wts[i]*beta

        alpha = math.log(1.0/beta)
        ans.append((x,alpha))
    
    
    return ans     





clas_list=[]
ftp=defaultdict(list)

def train_all(fea_img,wts):
    
    
    print "Training Weak Classifers ................"
    
    pos_wt, neg_wt=0,0
    global labels
    global ftp
    
    
    for i in range(len(wts)):
        if labels[i]==1:
            pos_wt+=wts[i]
        else:
            neg_wt+=wts[i]
    res=[]
    classifiers=[]

    for i in range(len(fea_img)):
        feature=fea_img[i]
        selected=sorted(zip(wts, feature, labels), key=lambda x: x[1])
        pos1, neg1=0,0
        posi_wts, negat_wts = 0, 0
        err_min, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
        for w, f, label in selected:
            error = min(negat_wts + pos_wt - posi_wts, posi_wts + neg_wt - negat_wts)
            if error < err_min:
                err_min = error
                best_feature = features[i]
                best_threshold = f
                best_polarity = 1 if pos1 > neg1 else -1
            if label == 1:
                pos1 += 1
                posi_wts += w
            else:
                neg1 += 1
                negat_wts += w
        ftp[features[i]]=(best_threshold, best_polarity)

        
        
    
    



faces_train = 'C:\\Users\\arifr\\.spyder\\violajones\\dataset\\trainset\\faces'
nonfaces_train = 'C:\\Users\\arifr\\.spyder\\violajones\\dataset\\trainset\\non-faces'

ft=[]
for pic in os.listdir(faces_train):
    ft.append(Image.open((os.path.join(faces_train, pic))))
ft_arr=im_arr(ft)

del ft[:]

nft=[]
for pic in os.listdir(nonfaces_train):
    nft.append(Image.open((os.path.join(nonfaces_train, pic))))
nft_arr=im_arr(nft)

del nft[:]

ft_int_arr=[]
for i in ft_arr:
    ft_int_arr.append(int_img(i))
nft_int_arr=[]   
for i in nft_arr:
    nft_int_arr.append(int_img(i))
    
print "creating Harr Features"
features=  features1_creation() + features2_creation() + features3_creation()+ features4_creation() + features5_creation()

print "Total of "+ str(len(features)) + " features are created"

x=len(nft_int_arr) + len(ft_int_arr)
fea_img=[[0]*x for _ in range(len(features))]

pos_wts=[1./2*len(ft_arr)]*len(ft_arr)
neg_wts=[1./2*len(nft_arr)]*len(nft_arr)
wts=pos_wts + neg_wts


train_res= [[0]*len(features) for i in range(x)]

labels1=[1]*len(ft_arr)
labels2=[-1]*(len(nft_arr))
labels= labels1+labels2



full_train =ft_int_arr+nft_int_arr




print "Applying features on training images"

for i,img in enumerate(full_train):
    
    for j,feature in enumerate(features):
        
        
        
        fea_val=0
        if feature[0]==1:
            
            fea_val=f1_val(img,feature[1], feature[2],feature[3],feature[4])
        elif feature[0]==2:
            fea_val=f2_val(img,feature[1], feature[2],feature[3],feature[4])
        elif feature[0]==3:
            fea_val=f3_val(img,feature[1], feature[2],feature[3],feature[4])
        elif feature[0]==4:
            fea_val=f4_val(img,feature[1], feature[2],feature[3],feature[4])
        elif feature[0]==5:
            fea_val=f5_val(img,feature[1], feature[2],feature[3],feature[4])
                
        fea_img[j][i]=fea_val
        
        
print "Features are applied on the images"

print "calling the function for adaboost to pick the strong classifier "

number_of_classifiers= 1

strong_clas= train(ft_int_arr,nft_int_arr,number_of_classifiers)

comp=0
strong_predict=[]
accu=0
z=ft_int_arr+nft_int_arr


print "printing strong class"
print strong_clas




print "sttrong classifer for error"


accu=0
sc_fp=0
sc_fn=0

for j,img in enumerate(z):
    cp=0
    alpha_sum=0
    
    
    for x  in strong_clas:
        alpha=x[1]
        
        feature ,ac,i,fp,fn = x[0][0],x[0][1],x[0][2], x[0][3], x[0][4]
        
        prediction=1 if fea_img[i][j]*ftp[features[i]][1]  < ftp[features[i]][0]*ftp[features[i]][1] else -1
        cp+= alpha*prediction
        
        alpha_sum+=alpha
        
    strong_predict= 1 if cp>=0.5*alpha_sum else -1
    if strong_predict==labels[j]:
        accu+=1
    if strong_predict==1 and labels[j]==-1:
        sc_fp+=1
    if strong_predict==-1 and labels[j]==1:
        sc_fn+=1
    
print "final Training Acuuracy,False positive, False_negative"
print float(accu)/len(z), float(sc_fp)/len(z),float(sc_fn)/len(z)




accu=0
sc_fp=0
sc_fn=0

#faces_test = 'C:\\Users\\arifr\\.spyder\\violajones\\dataset\\testset-copy\\faces' 
#nonfaces_test = 'C:\\Users\\arifr\\.spyder\\violajones\\dataset\\testset-copy\\non-faces'


faces_test = 'C:\\Users\\arifr\\.spyder\\violajones\\dataset\\testset\\faces' 
nonfaces_test = 'C:\\Users\\arifr\\.spyder\\violajones\\dataset\\testset\\non-faces'



def test(ft_arr,nft_arr):

    global test_labels
    global featest_img
    #global train_res
    #global wts                              #new
    global features  

    
    full_train= ft_arr+nft_arr
    
    print "working on testing set"

    
    for i,img in enumerate(full_train):
        for j,feature in enumerate(features):
            fea_val=0
            if feature[0]==1:
                fea_val=f1_val(img,feature[1], feature[2],feature[3],feature[4])
            elif feature[0]==2:
                fea_val=f2_val(img,feature[1], feature[2],feature[3],feature[4])
            elif feature[0]==3:
                fea_val=f3_val(img,feature[1], feature[2],feature[3],feature[4])
            elif feature[0]==4:
                fea_val=f4_val(img,feature[1], feature[2],feature[3],feature[4])
            elif feature[0]==5:
                fea_val=f5_val(img,feature[1], feature[2],feature[3],feature[4])
                
            featest_img[j][i]=fea_val 
            
    



ftest=[]
for pic in os.listdir(faces_test):
    ftest.append(Image.open((os.path.join(faces_test, pic))))
ftest_arr=im_arr(ftest)

del ftest[:]

nftest=[]
for pic in os.listdir(nonfaces_test):
    nftest.append(Image.open((os.path.join(nonfaces_test, pic))))
nftest_arr=im_arr(nftest)

del nftest[:]

ftest_int_arr=[]
for i in ftest_arr:
    ftest_int_arr.append(int_img(i))
nftest_int_arr=[]   
for i in nftest_arr:
    nftest_int_arr.append(int_img(i))

x=len(nftest_int_arr) + len(ftest_int_arr)
featest_img=[[0]*x for _ in range(len(features))]



test_labels1=[1]*len(ftest_arr)
test_labels2=[-1]*(len(nftest_arr))
test_labels= test_labels1+test_labels2

test(ftest_int_arr,nftest_int_arr)
comp=0
strong_predict=[]
accu=0
z1=ftest_int_arr+nftest_int_arr


sc_fp=0
sc_fn=0

print "applying the strong classifier "
for j,img in enumerate(z1):
    
    
    cp=0
    alpha_sum=0
    
    
    for x  in strong_clas:
        alpha=x[1]
        
        feature ,ac,i,fp,fn = x[0][0],x[0][1],x[0][2], x[0][3], x[0][4]
        
        prediction=1 if featest_img[i][j]*ftp[features[i]][1]  < ftp[features[i]][0]*ftp[features[i]][1] else -1
        cp+= alpha*prediction
        
        alpha_sum+=alpha
        
    strong_predict= 1 if cp>=0.5*alpha_sum else -1

    if strong_predict==test_labels[j]:
        accu+=1
    if strong_predict==1 and test_labels[j]==-1:
        sc_fp+=1
    if strong_predict==-1 and test_labels[j]==1:
        sc_fn+=1

print "Final Acuracy, Final False Positive,  Final False Negative of the strong classifier"
print float(accu)/len(z1), float(sc_fp)/len(z1), float(sc_fn)/len(z1)





    










    

