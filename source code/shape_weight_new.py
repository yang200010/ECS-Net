
import math
import os

import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def find_closest_point(points, target):
    closest_point = None
    closest_distance = float("inf")
    for point in points:
        distance = calculate_distance(point, target)
        if distance < closest_distance:
            closest_distance = distance
            closest_point = point
    return closest_point, closest_distance

def get_element_position(matrix, element):
    indices = np.where(matrix == element)
    positions = list(zip(indices[0], indices[1]))
    return positions

def get_skel(bin_img):
    skel, distance = morphology.medial_axis(bin_img, return_distance=True)
    return skel, distance


def get_shape(thresh, distance1, distance2, blackhat, h,w,nm,nb,nf):
    shape_weight_map = np.zeros((h, w))
    label = np.zeros((h, w))
    for ii in range(h):
        for jj in range(w):
            if thresh[ii][jj] == 0:  # background
                if blackhat[ii][jj] == 1:  # gap
                    label[ii][jj] = 1
                    shape_weight_map[ii][jj] = 4 * (nm / nb + nm / nf) 
                else:  # background
                    label[ii][jj] = 0
                    shape_weight_map[ii][jj] = nm / nb + nm / nf * (1 - min(distance1[ii][jj] / 10, 1))  # 计算距离，计算对应的权重
            else:  # instance
                if distance2[ii][jj] > 1:  # instance
                    label[ii][jj] = 2
                    if shape_weight_map[ii][jj] == 0:  
                        shape_weight_map[ii][jj] = nm / nf + nm / nf * math.exp(distance2[ii][jj] / 25)  # 计算距离，计算对应的权重
                else:  # edge
                    label[ii][jj] = 3
                    if shape_weight_map[ii][jj] == 0:
                        a = min(ii, 10)
                        b = min(jj, 10)
                        c = min(10, h - ii)
                        d = min(10, w - jj)
                        dis = np.zeros((h, w))
                        dis[(ii - a):(ii + c), (jj - b):(jj + d)] = (dist_on_skel2)[(ii - a):(ii + c),
                                                                    (jj - b):(jj + d)]
                        positions = get_element_position(dis, 1)
                        cloest_point, closest_distance = find_closest_point(positions, (ii, jj))  # 计算边缘点到中轴线上的距离
                        if cloest_point:
                            i, j = cloest_point[0], cloest_point[1]
                            if shape_weight_map[i][j] != 0:
                                shape_weight_map[ii][jj] = nm / nf + shape_weight_map[i][j] * (
                                            1 - min(closest_distance / 10, 1))
                            else:
                                shape_weight_map[i][j] = nm / nf + nm / nf * math.exp(
                                    distance2[i][j] / 25)  
                                shape_weight_map[ii][jj] = nm / nf + shape_weight_map[i][j] * (
                                            1 - min(closest_distance / 10, 1))
                        else:
                            shape_weight_map[ii][jj] = nm / nf
    return label, shape_weight_map


save1 = "./data/shape/"
save2 = "./data/label4/"
img_paths = glob.glob("./*.png")
for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, 0)
    thresh[thresh == 255] = 1
    h, w = thresh.shape

    shape_weight_map = np.zeros((h, w))
    label = np.zeros((h, w))


    dist_on_skel1, distance1 = get_skel(thresh)


    #indices_one = thresh == 1
    #indices_zero = thresh == 0
    #thresh[indices_one] = 0  # replacing 1s with 0s
    #thresh[indices_zero] = 1  # replacing 0s with 1
    dist_on_skel2, distance2 = get_skel(thresh)

    nb = h * w - np.sum(thresh) 
    nf = np.sum(thresh
    nm = min(nb, nf) 

    kernel = np.ones((7, 7), np.uint8)  
    blackhat = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)  
    for pp in range(4):
        for p in range(4):
            thresh1 = thresh[int(h / 4) * p:int(h / 4) * (p + 1), int(h / 4) * pp:int(h / 4) * (pp + 1)]
            distance11 = distance1[int(h / 4) * p:int(h / 4) * (p + 1), int(h / 4) * pp:int(h / 4) * (pp + 1)]
            distance21 = distance2[int(h / 4) * p:int(h / 4) * (p + 1), int(h / 4) * pp:int(h / 4) * (pp + 1)]
            blackhat1 = blackhat[int(h / 4) * p:int(h / 4) * (p + 1), int(h / 4) * pp:int(h / 4) * (pp + 1)]
            h1 = int(h / 4)
            w1 = int(h / 4)
            label1, shape_weight_map1 = get_shape(thresh1, distance11, distance21, blackhat1, h1, w1, nm, nb, nf)
            shape_weight_map[int(h / 4) * p:int(h / 4) * (p + 1),
            int(h / 4) * pp:int(h / 4) * (pp + 1)] = shape_weight_map1
            label[int(h / 4) * p:int(h / 4) * (p + 1), int(h / 4) * pp:int(h / 4) * (pp + 1)] = label1

   
    import matplotlib.pyplot as plt
    save_path1 = os.path.join(save1, img_path.split("/")[-1])
    save_path2 = os.path.join(save2, img_path.split("/")[-1])
    plt.imsave(save_path1, shape_weight_map, cmap='rainbow')
    plt.imsave(save_path2, label)






