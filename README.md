# AARC: Simple Sign Language Interpreter

A sign language interpreter that works on the basic principle of image pre-processing, KANE feature extraction and cosine distance matching.

In this project, we aim towards analyzing and recognizing various alphabets from a database of sign images. There is one image corresponding to the American Sign Language (ASL) gesture of one alphabet.

## Prerequisites

Before running this project, make sure you have the following dependencies:
- Python3
- OpenCV
- pip
- pickle

Now, using `pip install` command, install the following dependencies:
- Numpy
- Scipy
- matplotlib

Alternatively, you can use the following command to install them:
`pip install -r requirements.txt`

## Running

### Setting up dataset

The cosine vectors of all the features extracted get stored in a file called "features.pck". 

This repo contains a "features.pck" file already, which has been trained on the files in the "Data" folder. If you would like to use the existing dataset, skip to [Predicting Characters](#predicting-characters).

If you would like to train the dataset yourself, place your image files in the "Data" folder (with the filename same as the label name).

Then run:
`python3 AARC.py 1`

### Predicting Characters

Place your input image file in the folder. For example, the filename is `myImage.jpg`.

Run: 
`python3 AARC.py myImage.jpg`

The predicted character is displayed on the prompt.

## Working

The input image undergoes the following processes:

### Skin Masking
Threshold values for skin are defined in HSV. The image is converted from RGB schema HSV. Everything outside the threshold values is nullified to black. The image is then converted to greyscale.

### Noise Removal
All pixels that remain are of skin colour. There may still be noise, which is removed using Difference of Blur technique. 

### Background Subtraction
There are three techniques that can be used:

1. MOG: 

It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It was introduced in the paper “An improved adaptive background mixture model for real-time tracking with shadow detection” by P. KadewTraKuPong and R. Bowden in 2001. It uses a method to model each background pixel by a mixture of K Gaussian distributions (K = 3 to 5). The weights of the mixture represent the time proportions that those colours stay in the scene. The probable background colours are the ones which stay longer and more static.\[1\]

2. MOG2:

It is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It is based on two papers by Z.Zivkovic, “Improved adaptive Gausian mixture model for background subtraction” in 2004 and “Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction” in 2006. One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel. (Remember, in last case, we took a K gaussian distributions throughout the algorithm). It provides better adaptibility to varying scenes due illumination changes etc.\[1\]

3. Skin Thresholding
All the remaining values of skin colour are converted to white while others are turned black.

The program uses Skin Thresholding by default.

### Edge Detection

There are two techniques that can be used:

1. Canny Edge Detection

Canny edge detection is a multi-step algorithm that can detect edges with noise supressed at the same time. Smooth the image with a Gaussian filter to reduce noise and unwanted details and textures.

2. Sobel Edge Detection

The Sobel operator performs a 2-D spatial gradient measurement on an image and so emphasizes regions of high spatial frequency that correspond to edges. Typically it is used to find the approximate absolute gradient magnitude at each point in an input grayscale image.

The program uses Canny Edge Detection by default.

### References

1. [Background Subtraction Techniques](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html)

## Contributors

This project was built by 

1. Adit Nair
2. Aditya Nair
3. Christopher Paralkar
4. Rishabh Nanawati (me)

 as a mini project, part of our course Image Processing. We included a user-friendly GUI that allowed multiple registered users private access, fed input as the user's webcam video stream and saved the predicted characters into a log file.

 We also tried improving our dataset by including multiple images of the same label under different lighting conditions and used neural networks to better predict the input image.

 ### Screenshots

![Various stages of Image Processing for the ASL gesture "B"](/Screenshots/B.png)

![Various stages of Image Processing for the ASL gesture "Y"](/Screenshots/Y.png)

![Our application in working](/Screenshots/GUI.png)



 ## Future Scope

 This project is aimed at novice developers, trying to figure out their way around OpenCV and image matching. 

 It can be improved upon to have multiple images under different lighting conditions for the same character, support for more characters and a better prediction algorithms (perhaps using machine learning, deep learning techniques).

 





