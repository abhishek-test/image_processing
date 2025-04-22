import cv2
import numpy as np
from timeit import default_timer as timer
from cap_from_youtube import cap_from_youtube

'''

'''


def equalizeHistogram(frame):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    channels = cv2.split(frame)
    eq_channels = []

    for ch in channels:
        eq_channels.append(clahe.apply(ch))

    eq_clahe_image = cv2.merge(eq_channels)

    return eq_clahe_image

def calculateHistogram(frame):

    bgr_planes = cv2.split(frame)

    b_hist = cv2.calcHist(bgr_planes, [0], None, [256], (0,256), accumulate=False)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [256], (0,256), accumulate=False)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [256], (0,256), accumulate=False)

    cv2.normalize(b_hist, b_hist, alpha=0, beta=360, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=360, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=360, norm_type=cv2.NORM_MINMAX)

    return [b_hist, g_hist, r_hist]

def drawHistogram(frame, bgr_hist):

    histSize = 256
    hist_w = 640
    hist_h = 360
    bin_w = int(round( hist_w/histSize ))

    histImage = np.ones((hist_h, hist_w, 3), dtype=np.uint8)

    b_hist = bgr_hist[0]
    g_hist = bgr_hist[1]
    r_hist = bgr_hist[2]

    for i in range(1, histSize):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ), 
            ( bin_w*(i), hist_h - int(b_hist[i]) ), ( 255, 0, 0), thickness=2)
    
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ), 
            ( bin_w*(i), hist_h - int(g_hist[i]) ), ( 0, 255, 0), thickness=2)
    
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ), 
            ( bin_w*(i), hist_h - int(r_hist[i]) ), ( 0, 0, 255), thickness=2)
        
    outImg = cv2.hconcat([frame, histImage])
        
    cv2.line(outImg, (1, 0), (1,478), (0,0,0), 2, 8)
        
    return outImg


cap = cv2.VideoCapture(0)

while True:

    start = timer()

    ret, frame = cap.read()   
    frame = cv2.resize(frame, (640, 360)) 

    equalizedFrame = equalizeHistogram(frame)

    bgr_hist = calculateHistogram(frame)
    origOutImg = drawHistogram(frame, bgr_hist)

    eq_bgr_hist = calculateHistogram(equalizedFrame)
    origOutImg_eq = drawHistogram(equalizedFrame, eq_bgr_hist)

    outputImg = cv2.vconcat([origOutImg, origOutImg_eq])

    cv2.line(outputImg, (640,0), (640, 719), (0,255,255), 3)
    cv2.line(outputImg, (0,360), (1279, 360), (0,255,255), 3)


    end = timer()      

    cv2.imshow("Output", outputImg)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

    if key == ord('p'):
        cv2.waitKey()

        