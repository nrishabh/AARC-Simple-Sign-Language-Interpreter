import sys
import pickle
import cv2
import os
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
import random
import numpy as np
import TrainDB as moduleTDB
import FeatureExtraction as moduleFE

args = sys.argv

if args[1]=="1":
        print("\nTraining data from the Data folder.")
        moduleTDB.batch_extractor('Data')
        print("\nFeatures extracted from dataset and stored in feature.pck file.")
else:
    img = cv2.imread(args[1])


    class Matcher(object):

        def __init__(self, pickled_db_path="features.pck"):
            with open(pickled_db_path, 'rb') as fp:
                self.data = pickle.load(fp)
            self.names = []
            self.matrix = []
            for k, v in self.data.items():
                self.names.append(k)
                self.matrix.append(v)
            self.matrix = np.array(self.matrix)
            self.names = np.array(self.names)

        def cos_cdist(self, vector):
            # getting cosine distance between search image and images database
            v = vector.reshape(1, -1)
            return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

        def match(self, image, topn=5):
            features = moduleFE.extract_features(image)
            img_distances = self.cos_cdist(features)
            # getting top 5 records
            nearest_ids = np.argsort(img_distances)[:topn].tolist()
            nearest_img_paths = self.names[nearest_ids].tolist()

            return nearest_img_paths, img_distances[nearest_ids].tolist()


    def show_img(path):
        plt.imshow(img)
        plt.show()
        
    def run():
        ma = Matcher('features.pck')
        names, match = ma.match(moduleTDB.create_boundaryImage(img), topn=1)
        curr = names[0]
        return curr[0].upper()

    def crop_image(image, x, y, width, height):
        return image[y:y + height, x:x + width]

    result = run()
    print("\nPredicted Label: "+str(result))