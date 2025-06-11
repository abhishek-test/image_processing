import cv2
import numpy as np
from timeit import default_timer as timer
from cap_from_youtube import cap_from_youtube


def calculateHistogram(bgr_plnes):

    b_hist = cv2.calcHist(bgr_planes, [0], None, [256], (0,256), accumulate=False)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [256], (0,256), accumulate=False)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [256], (0,256), accumulate=False)

    cv2.normalize(b_hist, b_hist, alpha=0, beta=360, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=360, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=360, norm_type=cv2.NORM_MINMAX)

    # test for git bash setup
    return [b_hist, g_hist, r_hist] 

def drawHistogram(bgr_hist):

    histSize = 256
    hist_w = 640
    hist_h = 360
    bin_w = int(round( hist_w/histSize ))

    histImage_blue  = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    histImage_green = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    histImage_red   = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    b_hist = bgr_hist[0]
    g_hist = bgr_hist[1]
    r_hist = bgr_hist[2]


    for i in range(1, histSize):
        cv2.line(histImage_blue, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ), 
                 ( bin_w*(i), hist_h - int(b_hist[i]) ), ( 255, 0, 0), thickness=2)
    
        cv2.line(histImage_green, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ), 
                 ( bin_w*(i), hist_h - int(g_hist[i]) ), ( 0, 255, 0), thickness=2)
    
        cv2.line(histImage_red, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ), 
                 ( bin_w*(i), hist_h - int(r_hist[i]) ), ( 0, 0, 255), thickness=2)
        
    return [histImage_blue, histImage_green, histImage_red]

def drawFinal(bgr_planes, outImgHist):

    blueOrigFrame  = cv2.cvtColor(bgr_planes[0], cv2.COLOR_GRAY2BGR)
    greenOrigFrame = cv2.cvtColor(bgr_planes[1], cv2.COLOR_GRAY2BGR)
    redOrigFrame   = cv2.cvtColor(bgr_planes[2], cv2.COLOR_GRAY2BGR)

    blueOut  = cv2.addWeighted(blueOrigFrame, 0.8, outImgHist[0], 0.8, 0)
    greenOut = cv2.addWeighted(greenOrigFrame, 0.8, outImgHist[1], 0.8, 0)
    redOut   = cv2.addWeighted(redOrigFrame, 0.8, outImgHist[2], 0.8, 0)

    frameOrig = cv2.merge(bgr_planes)

    horconcat_1 = cv2.hconcat([frameOrig, blueOut])
    horconcat_2 = cv2.hconcat([greenOut, redOut])
    finalImage = cv2.vconcat([horconcat_1, horconcat_2])

    cv2.line(finalImage, (640,0), (640,719), (0,255,255), 2)
    cv2.line(finalImage, (0,360), (1279,360), (0,255,255), 2)

    return finalImage


youtube_url = 'https://www.youtube.com/watch?v=LT-oNf3A7IU'
#cap = cap_from_youtube(youtube_url)
cap = cv2.VideoCapture(0)

while True:

    start = timer()

    ret, frame = cap.read()    
    frame = cv2.resize(frame, (640, 360))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)

    if not ret:
        print("Error in reading video")
        continue

    bgr_planes = list(cv2.split(frame))

    bgr_hist = calculateHistogram(bgr_planes)
    outImgHist = drawHistogram(bgr_hist)
    outputImg = drawFinal(bgr_planes, outImgHist)

    end = timer()      

    cv2.imshow("Output", outputImg)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

    if key == ord('p'):
        cv2.waitKey()