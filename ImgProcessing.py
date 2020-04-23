import cv2
import numpy as np
import sys

# args = sys.argv
# imgInput = cv2.imread(args[1])

def doBackgroundSubtraction(imgInput, optAlgo=3):
    if (optAlgo==1):
        algo = cv2.createBackgroundSubtractorMOG2() # Using MOG2
    elif (optAlgo==2):
        algo =cv2. createBackgroundSubtractorMOG() # Using GMG
    elif (optAlgo==3):
        # Using our own code for thresholding
        h,w = imgInput.shape[:2]
        tempImage = cv2.cvtColor(imgInput, cv2.COLOR_HSV2BGR)  # Convert image from HSV to BGR format
        bw_image = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)  # Convert image from BGR to gray format
        bw_image = cv2.GaussianBlur(bw_image,(5,5),0)  # Highlight the main object
        threshold = 1
        for i in range(h):
            for j in range(w):
                if bw_image[i][j] > threshold:
                    bw_image[i][j] = 0
                else:
                    bw_image[i][j] = 255
        return bw_image
    return algo.apply(imgInput)

def doSkinMasking(imgInput):
    # print(str(imgInput))
    imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2HSV)
    bwInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)
    
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(imgInput, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    #cv2.imshow("masked",skinMask)
    
    skinMask = cv2.medianBlur(skinMask, 5)
    
    skin = cv2.bitwise_and(bwInput, bwInput, mask = skinMask)

    return skin

def doNoiseRemoval(imgInput, skinMask):
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(imgInput, imgInput, mask = skinMask)
    imgInput = cv2.addWeighted(imgInput,1.5,skin,-0.5,0)
    skin = cv2.bitwise_and(imgInput, imgInput, mask = skinMask)

    return skin

def doEdgeDetection(imgInput, optAlgo=1):
    if (optAlgo==1):
        return cv2.Canny(imgInput,60,60) #Canny Edge
    elif (optAlgo==2):
        return cv2.Sobel(imgInput, -1, 1, 1, 3) #Solbel Edge

def doDisplay(title, imgInput):
    #Function to display outputs
    cv2.imshow(title, imgInput)
    k = cv2.waitKey(0)

# print("\nStarting image outputs. \nKeep pressing a key in order to show new window.")
# doDisplay("Input Image", imgInput)

# img1 = doSkinMasking(imgInput)
# doDisplay("Skin Masked", img1)

# img2 = doNoiseRemoval(imgInput, img1)
# doDisplay("Noise Removed", img2)

# img3 = doBackgroundSubtraction(img2, 1)
# doDisplay("Background Subtraction", img3)

# img4 = doEdgeDetection(img3, 1)
# doDisplay("Edges detected: Sobel", img4)