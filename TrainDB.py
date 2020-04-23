import pickle
import random
import os
import cv2
import ImgProcessing as moduleIP
import FeatureExtraction as moduleFE

# imgInput = cv2.imread(img)

def create_boundaryImage(img):
    # shower("Input Image", imgInput)

    img1 = moduleIP.doSkinMasking(img)
    # shower("Skin Masked", img1)

    img2 = moduleIP.doNoiseRemoval(img, img1)
    # shower("Noise Removed", img2)

    img3 = moduleIP.doBackgroundSubtraction(img2)
    # shower("Background Subtraction MOG2", img3)

    img4 = moduleIP.doEdgeDetection(img3)
    # shower("Edges detected", img4)

    return img4

def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = moduleFE.extract_features(create_boundaryImage(cv2.imread(f)))
    
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)

# batch_extractor('Data')