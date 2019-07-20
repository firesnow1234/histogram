# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:52:39 2019

@author: Yuki
"""
import numpy as np
import cv2
import time
import copy
from math import isnan, isinf
from PIL import Image
import matplotlib.pyplot as plt
from myRansac import *
import math
import scipy.io as io
from scipy.io import loadmat 
import matlab.engine
from ransac import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from planeFit import planefit


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold


def project(p_in, T):
    """
    激光雷达投影矩阵变换，将点云投影到相机坐标系
    out: 投影结果
    """
    #dimension of data and projection matrix
    dim_norm = T.shape[0]
    #dim_proj = T.shape[1]
    
    # do transformation in homogenuous coordinates
    p2_in = p_in        
    p2_out = np.transpose(np.dot(T, np.transpose(p2_in)))
    
    #normalize homogeneous coordinates
    temp1 = p2_out[:, 0:dim_norm-1]
    temp2 = p2_out[:, dim_norm-1]
    temp3 = np.ones((1, dim_norm-1))    
    temp4 = temp2.reshape(len(p2_in), 1)
    
    p_out = temp1 / np.dot(temp4, temp3)
    
    return p_out



def lidar2img(lidar_index_cor2, velo_img, camera2, finalProp, pro_lidar, prop_threshold):
    """
    激光雷达到图像的投影显示  
    out: 返回在图像中的激光点（路面）
    """
    dis = 4  #前方激光点 x > 4
    x = []   
    y = []
    color = []
    lidarTmp = []
    for idx in range(len(lidar_index_cor2)):
        for jdx in range(len(lidar_index_cor2[0])):
            lidarIdx2 = int(lidar_index_cor2[idx, jdx])
            newPoint = velo_img[lidarIdx2, :]           
            if newPoint[1]>0 and newPoint[1]<camera2.shape[0]-1 and newPoint[0]>0 and newPoint[0]<camera2.shape[1]-1 and pro_lidar[lidarIdx2,0] > 4 and finalProp[idx][jdx] > prop_threshold :
                x.append(int(newPoint[0]))
                y.append(int(newPoint[1]))   
                color.append(64*dis // pro_lidar[lidarIdx2,0])
                lidarTmp.append(pro_lidar[lidarIdx2, :])
    
#    plt.figure()
#    plt.imshow(camera2)
#    plt.scatter(x, y, c=color, cmap=plt.cm.jet, marker='.', s=0.5)
#    plt.show()
#    plt.figure()
#    #plt.title('Road Curb',color='blue')
#    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16.0, 9.0) #dpi = 300, output = 700*700 pixels
    plt.imshow(camera2)
    plt.scatter(x, y, c=color, cmap=plt.cm.jet, marker=',', s=2)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    #fig.savefig("filename2.png", format='png', transparent=True, dpi=30, pad_inches = 0)
    #plt.scatter(x2, y2, c=color2, cmap=plt.cm.jet, marker='.', s=0.5)
    plt.show()
        
    return lidarTmp

   


# ==================================================================================================
#                                           Preprocessing
# ==================================================================================================
def preProcessing(s):
    """
    激光点云数据预处理
    out: 激光点云前方数据, 前方数据的 1/x 视差图
    """
    
    s[:,3]=1   
    # new computation
    index = np.arange(0, len(s),int(len(s)/64))     #每一条激光雷达的起始位置
    lidar_image_dis_temp = s.reshape(64,1737,4)    
    lidar_image_x = lidar_image_dis_temp[:,:,0]

    lidar_index_correspondance  = np.zeros((lidar_image_x.shape[0], lidar_image_x.shape[1]))   # 对应下标
    leftView, rightView = 0, 0
    for i in range(lidar_image_x.shape[1]):
        if lidar_image_x[31][i] > 5 and not isnan(lidar_image_x[31][i]):
            leftView = i
            break
    for i in range(lidar_image_x.shape[1]-1, 0, -1):
        if lidar_image_x[31][i] > 5 and not isnan(lidar_image_x[31][i]):
            rightView = i
            break
    for i in range(len(index) -1):  # 64
        for j in range(index[i], index[i+1]):  # 1737
            lidar_index_correspondance[i,j%1737] = j
    
    lidar_image_x[np.where(np.isnan(lidar_image_x))]= 0    

    plt.imshow(lidar_image_x, cmap=plt.cm.jet)
    plt.show()
    
    for i in range(lidar_image_x.shape[0]):
      for j in range(lidar_image_x.shape[1]):
        if lidar_image_x[i,j] < 0.001:
          lidar_image_x[i,j] = 0
        else:
          lidar_image_x[i,j]= 1 / lidar_image_x[i,j] 
    
    lidar_image_x[np.where(lidar_image_x ==np.Inf)] = 0
    
    lidar_image_front = lidar_image_x[:,leftView: rightView]   
    lidar_index_cor2 = lidar_index_correspondance[:, leftView: rightView]   
    
    lidar_image_front3 = np.zeros((lidar_image_front.shape[0], lidar_image_front.shape[1]))
    lidar_index_cor3 = np.zeros((lidar_index_cor2.shape[0], lidar_index_cor2.shape[1]))
    for i in range(int(lidar_image_front3.shape[1])):
       lidar_image_front3[:,i] = lidar_image_front[:, lidar_image_front.shape[1]-i-1]
       lidar_index_cor3[:, i] = lidar_index_cor2[:, lidar_index_cor2.shape[1] -i-1] 
    
    frontLidar = np.zeros((lidar_index_cor3.shape[0], lidar_index_cor3.shape[1], 3 ))  # 前方激光雷达数据xyz
    for i in range(lidar_index_cor3.shape[0]):
        for j in range(lidar_index_cor3.shape[1]):
            frontLidar[i][j] = lidar2[int(lidar_index_cor3[i][j]),:3] 
            

    lidar_image_front[np.where(np.isinf(lidar_image_front))] = 0  
    plt.figure()
    plt.imshow(lidar_image_front, cmap=plt.cm.jet)    
    plt.title('front')
    plt.figure()
    plt.imshow(lidar_image_front3, cmap=plt.cm.jet)    
    plt.title('lidar_image_front3')
    
    return frontLidar, lidar_image_front3, lidar_index_cor3
    
  
    
def histogram(lidar_image_front3):
    """
    Histogram 函数
    In: 前方的激光雷达点 
    Return
    """

    histogram = np.zeros((64,1000))
    for i in range(lidar_image_front3.shape[0]):
      for j in range(lidar_image_front3.shape[1]):
        indhis = int(lidar_image_front3[i,j]*1000)
        if indhis < 1000:
            histogram[i,indhis] += 1
    
    plt.figure()
    plt.imshow(histogram)
    plt.show()
    
    #二值化
    histArray =[]
    for x in range(30, 64):
        for y in range(1, 1000):
            if histogram[x][y] > 20:
                histArray.append([y,x])
                
    histPoints = np.array(histArray) 
    hist = []
    for i in range(len(histPoints)):
        hist.append([histPoints[i,0],histPoints[i,1]])
    
    
    #ransac直线拟合 y=mx+b
    m,b = plot_best_fit(hist)
    
    #路面分割的超参数
    alpha = 0.5
    beta = 1.2
    roadSegment = np.zeros([len(lidar_image_front3), len(lidar_image_front3[0])])
    
    # i: 激光扫描线,  j: 激光雷达获取图的前方视野点
    for i in range(len(lidar_image_front3)):
        for j in range(len(lidar_image_front3[0])):
            light = int(lidar_image_front3[i,j] / 0.001)
            if light > 1000:
                light = 0
            #case1: water
            if(lidar_image_front3[i,j] == 0 and i <= m*light + beta*b):
                roadSegment[i][j] = 0
            # case2： posetive obstacle
            elif(i <  m*light + alpha*b):
                roadSegment[i][j] = 1
            #case3: negative
            elif(i > m*light + beta*b):
                roadSegment[i][j] = 2
            #case4: road line
            elif(i >= m*light + alpha*b and i <= m*light + beta*b):
                roadSegment[i][j] = 3
    
    #print('totally cost',time_end-time_start)
    #归一化卷积核滤波
    #finalMap=cv2.blur(roadSegment,(5,5))
    
    '''
    plt.figure()            
    plt.imshow(roadSegment,cmap=plt.cm.jet)   
    plt.title('Road Segmentation')
    '''
        
    roadProp = np.zeros([len(lidar_image_front3), len(lidar_image_front3[0])])
    
    # i: 激光扫描线,  j: 激光雷达获取图的前方视野点
    #点（a，b）到直线Ax+By+C=0的距离为d=|Aa+Bb+C|/√(A^2+B^2)
    #直线 mx - y + b = 0
    maxDist = 0
    minDist = 0
    for i in range(len(lidar_image_front3)):
        for j in range(len(lidar_image_front3[0])):
            light = int(lidar_image_front3[i,j] / 0.001)
            if light == 0:
                continue
            dist = abs(m * light - i + b)/((-1)*(-1) + m * m)**0.5
            maxDist = max(maxDist, dist)
            minDist = min(minDist, dist)
    
    maxDist = 16 
    for i in range(len(lidar_image_front3)):
        for j in range(len(lidar_image_front3[0])):
            light = int(lidar_image_front3[i,j] / 0.001)
            if light == 0:
                continue
            dist = abs(m * light - i + b)/((-1)*(-1) + m * m)**0.5
            if dist > 16:
                roadProp[i,j] =  0
                continue
            roadProp[i,j] =  1 - (dist / (maxDist - minDist))
                  
    return roadProp



'''
###
### Main 函数
###
'''

if __name__ == '__main__':      
    lidarName = "..\\20190519\\1545204862.65.txt"
    camera2 = cv2.imread("..\\20190519\\1545204862.68.png")
    velo2imgFile = 'velo2img.txt'
    
    lidar1 = np.loadtxt(lidarName)
    velo2imgTxT = np.loadtxt(velo2imgFile)
    
    tempLidar =np.zeros((lidar1.shape[0],lidar1.shape[1]))
    # 每条line有1737个点
    for i in range(64):
        tempLidar[1737*i : 1737*i+1737, :] = lidar1[lidar1.shape[0]-i: 0: -64, :] #lidar[i:lidar.shape[0]:64,:]  
        
    lidar = tempLidar
    lidar2 =lidar[:,:4]
    
    # imshow project 
    pro_lidar = lidar[: ,:4]
    pro_lidar[:,3] = 1
    
    
    
    velo_img = project(pro_lidar, velo2imgTxT)       
    s = copy.deepcopy(lidar2) 
    frontLidar, lidar_image_front3, lidar_index_cor3 = preProcessing(s)
    
    roadProp = histogram(lidar_image_front3)             # histogram 计算道路概率   
    
    prop_threshold = 0.5  #阈值0~1 判断点是道路点还是非道路点
    lidarTmp = lidar2img(lidar_index_cor3, velo_img, camera2,  roadProp, pro_lidar, prop_threshold) 
    
    roadArray = np.array(lidarTmp)
   
    
    
    '''
    ### RANSAC
    ### 拟合平面，筛选点
    ###
    '''    
    def plot_plane(a, b, c, d):
        xx, yy = np.mgrid[:30, :30]
        return xx, yy, (-d - a * xx - b * yy) / c
    xyzs = roadArray[:,:3]
    n = len(xyzs)
    max_iterations = 10000
    goal_inliers = n * 0.3   

    # RANSAC
    m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    
    
    # 筛选与平面高度差小于 D m 的 点
    m1 = m.reshape(1,4)
    mask=abs(np.dot(m1,roadArray.T))    
    idx = np.where(mask < 0.15)    
    dist_array = roadArray[idx[1][:]]
    #io.savemat('dist_array', {'road_lidar': dist_array})    # 最后拟合曲面的点
    
    
    '''
    ### 最小二乘拟合曲面
    '''
    plane_args = planefit(dist_array[:,:3])
    
    
    
    
    '''
    ###
    ### 栅格化， 计算高程差
    ###
    '''
    
    Grid = []
    length = 60   # 0 ~ 60m
    width = 40    # -20 ~ 20 m
    cell_size = 0.2  # 0.2m
    for i in range(int(length*width / (cell_size**2))):
        Grid.append([])
    
    for i in range(len(pro_lidar)):
        lidar_i = pro_lidar[i]
        if 0<lidar_i[0]<60 and -20<lidar_i[1]<20:
            idx = int(lidar_i[0]/0.2) * 200 + int((lidar_i[1]+20) /0.2)
            Grid[idx].append(lidar_i) 
      
    dis = 4
    x = []
    y = []
    color = []
    lidar_grid = []
    for i in range(len(velo_img)):
        newPoint = velo_img[i,:]
        lidar_i = pro_lidar[i,:3]
        if 0<lidar_i[0]<60 and -20<lidar_i[1]<20:
            idx = int(lidar_i[0]/0.2) * 200 + int((lidar_i[1]+20) /0.2)
            lidar_cell = np.array(Grid[idx])
            dist_cell = abs(np.dot(m1,lidar_cell.T))
            if np.max(dist_cell) < 0.2:  #and np.max(lidar_cell[:,2]) - np.min(lidar_cell[:,2]) < 0.2:   # 计算到ransac平面距离 以及 高程差 
                if newPoint[1]>0 and newPoint[1] < camera2.shape[0]-1 and newPoint[0]>0 and newPoint[0]<camera2.shape[1]-1 and pro_lidar[i,0] > dis and len(Grid[idx]) > 0 :
                    x.append(int(newPoint[0]))
                    y.append(int(newPoint[1]))
                    color.append(64*dis // pro_lidar[i,0])   
                    lidar_grid.append(lidar_i)

    plt.figure()
    plt.title('Altitude difference',color='blue')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16.0,9.0) #dpi = 300, output = 700*700 pixels
    plt.imshow(camera2)
    plt.scatter(x, y, c=color, cmap=plt.cm.jet, marker=',', s=2)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.savefig("filename2.png", format='png', transparent=True, dpi=30, pad_inches = 0)
    #plt.scatter(x2, y2, c=color2, cmap=plt.cm.jet, marker='.', s=0.5)
    plt.show()
    
    x10, y10 = [], []
    for it in pro_lidar:
        if 0<it[0] < 60 and -20<it[1]<20:
            x10.append(it[0])
            y10.append(it[1])
    
    plt.scatter(y10, x10, c='black', marker=',', s=1)
    plt.grid()
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.axis([-20,20,0,60]) 
    plt.gca().invert_xaxis() 
    plt.show()
    

                       
    
    '''
    ### 显示
    '''
        
    mat_data = loadmat('mydata.mat')
    lidar_mat = mat_data["result"]
    clo_b = np.ones(len(lidar_mat))
    
    p0,p1,p2,p3,p4,p5 = plane_args[5], plane_args[3], plane_args[4], plane_args[0], plane_args[1], plane_args[2]
    # 计算点到二次曲面的距离 
    def compute_dist2plane(xyz):
        # comp_z = A + B*x + C*y + D*x.*x + E*x.*y + F*y.*y;       
        x, y, z = xyz[0], xyz[1], xyz[2]
        comp_z = p0 + p1*x + p2*y + p3*x*x + p4*x*y + p5*y*y
        dist = z - comp_z
        return dist
    
    dist2plane_all = list(map(compute_dist2plane, list(pro_lidar)))
    
    pointStack = np.c_[pro_lidar, dist2plane_all]    #对原始点云数据进行 列叠加

    np.savetxt('finalStk.txt', pointStack, fmt='%0.6f')
    
    lidar_mat2 = np.c_[lidar_mat, clo_b]
    x2 = []
    y2 = []
    color2 = []
    velo_img2 = project(lidar_mat, velo2imgTxT)
    lidar_road2 = []
    final_ground = np.zeros((len(pro_lidar), 3))
    
    for i in range(len(velo_img2)):
        newPoint = velo_img2[i,:]
        lidar_i = lidar_mat2[i,:3]
        if 0<lidar_i[0]<60 and -20<lidar_i[1]<20:
            idx = int(lidar_i[0]/0.2) * 200 + int((lidar_i[1]+20) /0.2)  # 栅格坐标
            lidar_cell = np.array(Grid[idx])
            if len(lidar_cell) > 0:  # 栅格内的点数 > 0
                dist_cell = abs(np.dot(m1, lidar_cell.T))
                if np.max(dist_cell) < 0.15 and np.max(lidar_cell[:,2]) - np.min(lidar_cell[:,2]) < 0.2:
                    if newPoint[1]>0 and newPoint[1] < camera2.shape[0]-1 and newPoint[0]>0 and newPoint[0]<camera2.shape[1]-1 and lidar_mat2[i,0] > dis:   
                        x2.append(int(newPoint[0]))
                        y2.append(int(newPoint[1]))
                        color2.append(64*dis // lidar_mat2[i,0])
                        lidar_road2.append(lidar_mat2[i,:3])
                      
    plt.figure()
    #plt.title('Final Result',color='blue')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16.0,9.0) #dpi = 300, output = 700*700 pixels
    plt.imshow(camera2)
    plt.scatter(x2, y2, c=color2, cmap=plt.cm.jet, marker=',', s=2)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.show()
    
     
    x1, y1 = [], []
    for it in lidar_road2:
        if 0<it[0] < 60 and -20<it[1]<20:
            x1.append(it[0])
            y1.append(it[1])
    
    plt.scatter(y1, x1, c='black', marker=',', s=2)
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.axis([-20,20,0,60]) 
    plt.gca().invert_xaxis() 
    plt.show()
     
    
  
    
        
    
    
    
        
    


    
        

    
    

    
    



        
    

    
   









