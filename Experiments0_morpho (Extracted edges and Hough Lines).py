#!/usr/bin/env python
# coding: utf-8

# # Edge Detection

# In[1]:


import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import os
from cv2.ximgproc import guidedFilter
from cv2.ximgproc import l0Smooth
import math
count = 0
path = "HoughLines/"


# In[2]:


def my_polygon(img,pts):
    line_type = 8
    # Create some points
    ppt = np.array(pts, np.int32)
    ppt = ppt.reshape((-1, 1, 2))
    img = cv2.fillPoly(img, [ppt], (0, 0, 0), line_type)
#    img = cv2.polylines(img, [ppt], True,(0, 0, 0), line_type)
    return img


# In[3]:


pts1 = [ [190,0],
        [195,475],
        [292,500],
        [292,0]     ]
pts2 = [ [0,0],
        [40,0],
        [40,500],
        [0,500]     ]

pts1 = [ [190,0],
        [195,644],
        [292,644],
        [292,0]     ]
pts2 = [ [0,0],
        [40,0],
        [40,644],
        [0,644]     ]


# In[4]:




# In[5]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def Gabor_filtering(gray, ksize=111, sigma=10, gamma=1.2, lamda=10, phi=0, theta=0):
    
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
        
    # filtering
    out = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    return out

# Use 6 Gabor filters with different angles to perform feature extraction on the image
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # define angle
    #As = [0, 45, 90, 135]
    As = [0, 1*np.pi/2]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, ksize=11, sigma=1.5, gamma=1.2, lamda=3, theta=A)
        
        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


# In[32]:


import cv2
import numpy as np

cap = cv2.VideoCapture("video_building.mp4")
# cap = cv2.VideoCapture("production ID_4434242.mp4")

  
# reads frames from a camera
suc, img = cap.read()
if suc==False:
    print("No frame!!!")
    exit()
w,h,c = img.shape
blank0 = np.zeros((w, h, c), dtype = "uint8")
Progress = blank0.copy()
r,g,b = 0,0,0
while True:
    if(b<255):
        b+=20
    if(b>=255):
        b=0
        if(g<255):
            g+=20
        if(g>=255):
            g=0
            if(r<255):
                r+=20
            if(r>=255):
                r=0


    suc, img = cap.read()
#     print(img.shape)
    if suc == False:
        break
#     w,h,c = img.shape
#     img = cv2.resize(img,(w//5,h//4))
#     difference = cv2.subtract(img, img0)

    img0 = img.copy()
    dst_img = img.copy()
#Smoothening
#     img = l0Smooth(img, kappa=1.0)
#     show_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

#Canny
#     img_canny = cv2.Canny(img,200,250, True)
#     dst_canny = cv2.Canny(dst,200,250, True)
#     show_dst_canny = cv2.Canny(show_dst,200,250, True)

#Guided Filter: seems like doing nothing
#    edges = guidedFilter(img,img1,50,100)    


#     img = cv2.imread("C:/temp/1.png")

#     img = cv2.GaussianBlur(img, (7, 7), 0)
#     img = cv2.medianBlur(img, 8)
#     img = cv2.bilateralFilter(img, 9, 75, 75)


    frame1 = img.copy()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#   blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Display an original image
#     cv2.imshow('Original', gray)

    # finds edges in the input image image and
    # marks them in the output map edges

    blur = cv2.GaussianBlur(gray,(3,3),0)

    #adpt hist
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    adpt_hist = clahe.apply(blur)

    #gabor filtering process
    img = frame1.astype(np.float32)

    out = Gabor_process(frame1)
    img_gb = cv2.add(out, gray)

    #canny edge ditection
    edges = cv2.Canny(img_gb,190,250)

    #closing
    kernelSize = (2,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    #opening 
    kernelSize = (2,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    opening_edges = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    edges = closing


# #     mean, std = cv2.meanStdDev(gray)
# #     gray = gray - mean
# #     cl1 = gray/std

# # # Adaptive Histogram
# # #     clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
# # #     print(type(clahe))
# # #     cl1 = clahe.apply(clahe)
# #     print(cl1.shape)
#     edges = cv2.Canny(gray, 200, 300,apertureSize = 3)


#     dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#     dst_edges = cv2.Canny(dst_gray, 150, 200,apertureSize = 3)
#     edges = my_polygon(edges,pts1)
#     edges = my_polygon(edges,pts2)


#HoughLines : return lines in polar cordinate system: which will need to be converted into
#             cartetian cordinate system to draw on image

# edges: Output of the edge detector.
# lines: A vector to store the coordinates of the start and end of the line.
# rho: The resolution parameter \rho in pixels.
# theta: The resolution of the parameter \theta in radians.
# threshold: The minimum number of intersecting points to detect a line.
#     rho = 5
#     Thetares = math.pi/2
#     Threshold = 1
#     minLineLength = 50
#     maxLineGap = 10
#     lines = cv2.HoughLines(edges,rho,Thetares,Threshold)
# #     print(type(lines))
#     try:
#         for line in lines:
#             if type(line)=='NoneType':
#                 continue
#             rho, theta = line[0]
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = rho * a
#             y0 = rho * b
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#             print(x1,y1,x2,y2)
#             cv2.line(img0, (x1, y1), (x2, y2), (0, 0, 255), 1)
#     except:
#         continue


#HoughLinesP: return lines in cartetian cordinate system    

#     lines = cv2.HoughLinesP(edges,Rres,Thetares,Threshold,minLineLength,maxLineGap)
#     print(lines)

    rho = 10
    Thetares = (math.pi)/2
    Threshold = 10
    minLineLength = 25
    maxLineGap = 5

    lines = cv2.HoughLinesP(edges,rho,Thetares,Threshold,minLineLength,maxLineGap)
    Hough_img = blank0.copy()
    try:
        for line in lines:
            x1,y1,x2,y2 = line[0]

            cv2.line(img0,(x1,y1),(x2,y2),(255,255,0),1)
            cv2.line(Hough_img,(x1,y1),(x2,y2),(255,255,0),1)
            color = (r,g,b)
            cv2.line(Progress,(x1,y1),(x2,y2),color,1)
    except:
        continue


#     dst_lines = cv2.HoughLinesP(dst_edges,rho,Thetares,Threshold,minLineLength,maxLineGap)
#     try:
#         for line in dst_lines:
#             x1,y1,x2,y2 = line[0]
#             cv2.line(dst_img,(x1,y1),(x2,y2),(0,255,0),1)
#     except:
#         continue


    #  lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, 30, 1)
#     try:
#         for line in lines[0]:
#             pt1 = (line[0],line[1])
#             pt2 = (line[2],line[3])
#             cv2.line(img0, pt1, pt2, (0,0,255), 3)
#     except:
#         continue

#     cv2.imwrite("C:/temp/2.png", img)

#     zeros = np.zeros(img.shape[:2], dtype="uint8")

#     final_edges = cv2.hconcat([img, dst])

#     print(edges)

#     edges = cv2.cvtColor(edges, cv2.CV_GRAY2RGB) 
    closing = cv2.bitwise_and(dst_img, dst_img, mask=closing)

#         final_img = cv2.hconcat([img,edges,Hough_img,img0,Progress])
#     print("img.shape:",img.shape)
#     print("Hough_img.shape",Hough_img.shape)
#     print("img0.shape",img0.shape)
#     print("closing.shape",closing.shape)

#     Hough_img = Hough_img.astype('float32')
#     img0 = img0.astype('float32')
#     dst_img = dst_img.astype('float32')
#     closing = closing.astype('float32')

    print("img.shape:",img[0,0,0].dtype)
    print("Hough_img.shape",Hough_img[0,0,0].dtype)
    print("img0.shape",img0[0,0,0].dtype)
    print("closing.shape",closing[0,0,0].dtype)

    #     edges = cv2.bitwise_and(dst_img, dst_img, mask=dst_edges)
    final_img = cv2.hconcat([dst_img,closing,Hough_img, img0])

#     final = cv2.vconcat([final_img,dst_final_img])
    cv2.imshow('final_img', final_img)
#     cv2.imshow('final_img0', img0)
#     cv2.imshow('final_edges', edges)  
#         time.sleep(0.01)
#         cv2.imshow('hough', Hough_img)
#         cv2.imshow('final',img0)
#         cv2.imshow('closing', closing)
#         cv2.waitKey(0)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
#     cv2.imwrite(path+str(count)+".png",img0)
    count+=1

#cap.destroyAllWindows()
cv2.destroyAllWindows()


# In[7]:


# import cv2 as cv
# from matplotlib import pyplot as plt

# image_path = '0.jpg'
# img = cv.imread(image_path)

# start = cv.getTickCount()
# dst = l0Smooth(img, kappa=2.0)
# end = cv.getTickCount()

# print('filter algorithm time is ', (end-start)/cv.getTickFrequency(), ' seconds.')

# show_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# plt.subplot(121), plt.imshow(show_img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# show_dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
# plt.subplot(122), plt.imshow(show_dst), plt.title('l0smooth')
# plt.xticks([]), plt.yticks([])
# plt.show()

