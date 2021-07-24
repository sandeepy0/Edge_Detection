DEBUG = False
color = (255,0,0)
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
canny_th_1 = 297
canny_th_2 = 327
GB_channel_size = 3

#Houghlines_1 parameters
rho1 = 10
th1 = 1
minLineLength1 = 25
maxLineGap1 =20

#Houghlines_2 parameters
rho2 = 10
th2 = 1
minLineLength2 = 1
maxLineGap2 =20


pts1 = [ [190,0],
        [195,475],
        [292,500],
        [292,0]     ]
pts2 = [ [0,0],
        [40,0],
        [40,500],
        [0,500]     ]

def my_polygon(img,pts):
    line_type = 8
    ppt = np.array(pts, np.int32)
    ppt = ppt.reshape((-1, 1, 2))
    img = cv2.fillPoly(img, [ppt], (0, 0, 0), line_type)
    return img

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
def nothing(x):
    pass

cap = cv2.VideoCapture("video_building.mp4")
# cap = cv2.VideoCapture("production ID_4434242.mp4")
cv2.namedWindow('Final_Frame')

#Create two sliders to control parameters respectively.
cv2.createTrackbar('canny_th_lower','Final_Frame',canny_th_1,1000,nothing) #canny lower threshold
cv2.createTrackbar('canny_th_upper','Final_Frame',canny_th_2,1000,nothing) #canny upper threshold
cv2.createTrackbar('Hough_rho_1','Final_Frame',rho1,50,nothing)
cv2.createTrackbar('Hough_rho_2','Final_Frame',rho2,50,nothing)
cv2.createTrackbar('Hough_1_th','Final_Frame',th1,100,nothing)
cv2.createTrackbar('Hough_2_th','Final_Frame',th2,100,nothing)
cv2.createTrackbar('minLineLength1','Final_Frame',minLineLength1,100,nothing)
cv2.createTrackbar('minLineLength2','Final_Frame',minLineLength2,100,nothing)
cv2.createTrackbar('maxLineGap1','Final_Frame',maxLineGap1,100,nothing)
cv2.createTrackbar('maxLineGap2','Final_Frame',maxLineGap2,100,nothing)

#cv2.createTrackbar('canny_th_upper','Final_Frame',canny_th_2,1000,nothing)
#cv2.createTrackbar('canny_th_upper','Final_Frame',canny_th_2,1000,nothing)
#cv2.createTrackbar('canny_th_upper','Final_Frame',canny_th_2,1000,nothing)


# reads frames from a camera
suc, img = cap.read()
if suc==False:
    print("No frame!!!")
    exit()
w,h,c = img.shape
blank0 = np.zeros((w, h, c), dtype = "uint8")
Progress = blank0.copy()
r,g,b = 0,0,0
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
while True:
#Changing color
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
    if suc == False:
        break
    while True:
        guidelines_for_Hough2 = np.zeros((w, h),dtype = "uint8")
        #taking parameters from Trackbar
        canny_th_1 = cv2.getTrackbarPos('canny_th_lower','Final_Frame')
        canny_th_2 = cv2.getTrackbarPos('canny_th_upper','Final_Frame')
        Hough_rho_1= cv2.getTrackbarPos('Hough_rho_1','Final_Frame')
        Hough_rho_2= cv2.getTrackbarPos('Hough_rho_2','Final_Frame')
        Hough_1_th = cv2.getTrackbarPos('Hough_1_th','Final_Frame')
        Hough_2_th = cv2.getTrackbarPos('Hough_2_th','Final_Frame')
        minLineLength1 = cv2.getTrackbarPos('minLineLength1','Final_Frame')
        minLineLength2 = cv2.getTrackbarPos('minLineLength2','Final_Frame')
        maxLineGap1 = cv2.getTrackbarPos('maxLineGap1','Final_Frame')
        maxLineGap2 = cv2.getTrackbarPos('maxLineGap2','Final_Frame')

  #      GB_channel_size=cv2.getTrackbarPos('GB_channel_size','Final_Frame')
        
        if(DEBUG):
            print(1)
        img0 = img.copy()
        dst_img = img.copy()
        frame1 = img.copy()
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        GB_channel = (GB_channel_size,GB_channel_size)
        blur = cv2.GaussianBlur(gray,GB_channel,0)
        
        if(DEBUG):    
            print(2)
        #adpt hist
        if(DEBUG):
            print("type(blur):",(blur[0,0].dtype))
        if(DEBUG):
            print("blur.shape:",blur.shape)
        adpt_hist = clahe.apply(blur)
        if(DEBUG):
            print(3)    
        #gabor filtering process
        img1 = frame1.astype(np.float32)
        out = Gabor_process(frame1)
        img_gb = cv2.add(out, gray)
        if(DEBUG):
            print(4)    
        #canny edge ditection
        edges = cv2.Canny(img_gb,canny_th_1,canny_th_2)
        if(DEBUG):    
            print(5) 
            
        #closing
        kernelSize = (2,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        if(DEBUG):
            print(6)    
        #opening 
        kernelSize = (2,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        opening_edges = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        
        edges = closing
        Thetares = (math.pi)/2
        if(DEBUG):
            print(7)  
        lines = cv2.HoughLinesP(edges,Hough_rho_1,Thetares,Hough_1_th,minLineLength1,maxLineGap1)
        Hough_img = blank0.copy()
        Hough_img1=blank0.copy()
        try:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(img0,(x1,y1),(x2,y2),(255,255,0),1)
                cv2.line(Hough_img,(x1,y1),(x2,y2),(255,255,0),1)
                dynamic_color = (r,g,b)
                cv2.line(Progress,(x1,y1),(x2,y2),dynamic_color,1)
                cv2.line(guidelines_for_Hough2,(x1,y1),(x2,y2),1)
                
        except:
            continue

        Thetares = (math.pi)/2
        lines = cv2.HoughLinesP(guidelines_for_Hough2,Hough_rho_2,Thetares,Hough_2_th,minLineLength2,maxLineGap2)
        try:
            for line in lines:
                x1,y1,x2,y2 = line[0]
 #               cv2.line(img0,(x1,y1),(x2,y2),(255,255,0),1)
  #              cv2.line(Hough_img,(x1,y1),(x2,y2),(255,255,0),1)
                color = (r,g,b)
                cv2.line(Hough_img1,(x1,y1),(x2,y2),color,1)
   #             cv2.line(guidelines_for_Hough2,(x1,y1),(x2,y2),1)
        except:
            continue
        
        closing = cv2.bitwise_and(dst_img, dst_img, mask=closing)
        if(DEBUG):
            print(8)    
            print("img.shape:",img1[0,0,0].dtype)
            print("Hough_img.shape",Hough_img[0,0,0].dtype)
            print("img0.shape",img0[0,0,0].dtype)
            print("closing.shape",closing[0,0,0].dtype)
            
        final_img = cv2.hconcat([dst_img,closing,Hough_img, img0,Progress, Hough_img1])
        cv2.imshow('Final_Frame', final_img)
                
        if cv2.waitKey(1)==ord('c'):
            break
    if cv2.waitKey(0)==ord('q'):
        break
#     cv2.imwrite(path+str(count)+".png",img0)
    count+=1
cap.release()
cv2.destroyAllWindows()