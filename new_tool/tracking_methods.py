import numpy as np
import KCFpy.kcftracker as kcf
import cv2

# Constants for the tracker
selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

autoselect_th = 0.8  # initialize the tracker or get the annotation when the
# confidence of the tracker goes below this score


def linear_annotation(imgs, gts, stride = 30):
    '''
    Function to interpolate the intermidiate annotations
    :param imgs: list of images
    :param gts: list of GT bounding boxes
    :return: list of predicted bboxes, list of annotations
    '''
    
    # initialize the tracker
    tracker = kcf.KCFTracker(True, False, False)  # hog, fixed_window,
    # multiscale
    # if you use hog feature, there will be a short pause after you draw a
    # first boundingbox, that is due to the use of Numba.
    tracker.init(gts[0], imgs[0])
    prediction = []
    for e, (i, b) in enumerate(zip(imgs, gts)):
        if e % stride == 0:
            tracker.init(b, i)
            boundingbox = map(int, b)
            prediction.append([boundingbox, False])
            continue
        boundingbox, val, _ = tracker.update(i)
        boundingbox = map(int, boundingbox)
        prediction.append([boundingbox, True])
        
    return prediction


def auto_select(imgs, gts, stride = 30):
    '''
    Function to interpolate the intermidiate annotations
    :param imgs: list of images
    :param gts: list of GT bounding boxes
    :return: list of predicted bboxes, list of annotations
    '''
    
    # initialize the tracker
    tracker = kcf.KCFTracker(True, False, False)  # hog, fixed_window,
    # multiscale
    # if you use hog feature, there will be a short pause after you draw a
    # first boundingbox, that is due to the use of Numba.
    tracker.init(gts[0], imgs[0])
    prediction = []
    for e, (i, b) in enumerate(zip(imgs, gts)):
        if e % stride == 0:
            tracker.init(b, i)
            boundingbox = map(int, b)
            prediction.append([boundingbox, False])
            continue
        boundingbox, val, _ = tracker.update(i)
        boundingbox = map(int, boundingbox)
        prediction.append([boundingbox, True])
        
    return prediction
