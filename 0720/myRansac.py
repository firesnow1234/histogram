# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:07:19 2018

@author: Yuki
"""
import random
import numpy as np
from matplotlib import pyplot as plt

# Magic Numbers
# Controls the inlier range
THRESHOLD = 0.1

# Finds random potential fit lines
def RANSAC(data):
    n = len(data)

    # Another magic number
    NUM_TRIALS = n // 2

    best_in_count = 0
    for i in range(0, NUM_TRIALS):
        r = random.sample(data, 2)
        r = np.array(r)

        # linear regression on two points will just give the line through both points
        m, b = lin_reg(r)

        # finds the line with the most inliers
        in_count = 0
        for j in data:
            # if the distance between the line and point is less than or equal to THRESHOLD it is an inlier
            if abs(j[1] - ((m * j[0]) + b)) <= THRESHOLD:
                in_count = in_count + 1
        # Tracks the best fit line so far
        if in_count > best_in_count:
            best_in_count = in_count
            best_m = m
            best_b = b

    # record both inliers and outliers to make end graph pretty
    in_line = []
    out_line = []
    for j in data:
        if abs(j[1] - ((best_m * j[0]) + best_b)) <= THRESHOLD:
            in_line.append(j)
        else:
            out_line.append(j)

    # returns two lists, inliers and outliers
    return in_line, out_line

# performs the linear regression as described on the assignment sheet
def lin_reg(data):
    n = float(len(data))
    x_sum = 0.0
    y_sum = 0.0

    # averages the x and y values
    for i in data:
        x_sum = x_sum + i[0]
        y_sum = y_sum + i[1]
    x_average = x_sum / n
    y_average = y_sum / n

    # initializes slope numerator and denominator
    # note denominator should not be zero with data
    m_numerator = 0.0
    m_denominator = 0.0

    # calculates the slope
    for i in data:
        m_numerator = m_numerator + ((i[0] - x_average)*(i[1] - y_average))
        m_denominator = m_denominator + ((i[0] - x_average)*(i[0] - x_average))
    m = m_numerator / m_denominator

    # finds the intercept
    b = y_average - (m * x_average)

    # returns slope and intercept
  
    return m, b

def plot_best_fit(data):

    # Get our inlier and outlier points
    in_line, out_line = RANSAC(data)

    # find the best fit line for inliers
    m, b = lin_reg(in_line)
    
    # This was the hardest part
    # Could not find a function that would make a non line segment so I just covered our domain
    # Admittedly with potential error on giant domains
    x_min = 100000.0
    x_max = -100000.0
    for i in data:
        if i[0] > x_max:
            x_max = i[0]
        if i[0] < x_min:
            x_min = i[0]
    domain = [x_min, x_max]
    line_points = [m * i + b for i in domain]
    line_points_top= [m * i + 0.5 * b for i in domain]
    line_points_bottom = [m * i + 1.2 * b for i in domain]
    
    # Plot the inliers as blue dots
    in_line = np.array(in_line)
    x, y = in_line.T
    #plt.scatter(x, y)

    # plot the outliers as red x's
    # if statement for if outliers is empty, which it is for the easy case
    if out_line != []:
        out_line = np.array(out_line)
        x, y = out_line.T
        #plt.scatter(x, y, s=30, c='black', marker='x')

    # plot our best fit line
    plt.plot(domain, line_points, '-',  c='black',  linewidth = '3')    
    #plt.plot(domain, line_points_bottom, '-', c='black')    
    #plt.plot(domain, line_points_top, '-',  c='black')    
    plt.gca().invert_yaxis()
    # show the plot
    


    #设置横纵坐标的名称以及对应字体格式
    font2 = {'family' : 'Times New Roman', 'weight' : 'normal',   'size'   : 15,    }

    #plt.title("Road-Line-Estimation")
    
  
    plt.xlabel(r'$\frac{1}{x}$', fontsize =20)
    plt.ylabel(r'$n$', fontsize=15)

    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.show()
    
    # return slope and intercept for answers
    print("m"+str(m)+"b"+str(b))
    return m, b

# ----------------------------------------------------------------------------------------------------
'''
data = []

with open('noisy_data_medium.txt') as file:
    # Creates 2D array to hold the data
    for l in file:
        data.append(l.split())

    # removes comma from first entry
    for i in data:
        i[0] = float(i[0][:-1])
        i[1] = float(i[1])

# function also returns slop and intercept should you want them
m, b = plot_best_fit(data)
print(m, b)
'''
