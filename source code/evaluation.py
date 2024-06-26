import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    # TP = ((SR==1)+(GT==1))==2
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR==0).byte()+(GT==1).byte())==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).byte()+(GT==0).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).byte()+(GT==1).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)
    #print(F1)

    return F1

def get_DICE(SR,GT):
    SR = SR
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    dice = 2 * float(corr) / (corr * corr + SR * SR)
    return dice

import numpy as np
from scipy import ndimage
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim==2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


import GeodisTK
def binary_hausdorff95(s, g, spacing=None,threshold=0.5):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2

    """

    s = s > threshold
    g = g == torch.max(g)
    s=s.squeeze(0).squeeze(0)
    g=g.squeeze(0).squeeze(0)

    s_edge = get_edge_points(s.cpu())
    g_edge = get_edge_points(g.cpu())

    image_dim = len(s.shape)

    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim, len(spacing))
    img = np.zeros_like(s.cpu())
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)

def get_IOU(S ,T, threshold=0.5):
    S = S > threshold
    T = T == torch.max(T)
    intersecion = np.multiply(S.cpu(), T.cpu())

    union = np.asarray(S.cpu() + T.cpu() > 0, np.float32)
    iou = float(intersecion.sum()) / float((union.sum() + 1e-10))
    return iou

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity

    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte()+GT.byte()).byte()==2)
    Union = torch.sum((SR+GT)>=1)


    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte()+GT.byte()).byte()==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

def get_PHD(SR,GT,tolerance=None):
    # PHD : Perceptual Hausdorff
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



