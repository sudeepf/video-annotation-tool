import cv2
import numpy as np


def plot_on_image(imgs, est, color = (0, 255, 255)):
    for i, (im, bbox) in enumerate(zip(imgs,est)):
        is_annot = bbox[1]
        bbox = bbox[0]
        if (is_annot):
            cv2.rectangle(im, (bbox[0], bbox[1]), (
                bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          color, 1)
        else:
            cv2.rectangle(im, (bbox[0], bbox[1]), (
                bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          color, 3)


def visualize_video(imgs, est_lin, est_auto, gt):

    plot_on_image(imgs, est_lin, (255, 0, 0))
    plot_on_image(imgs, est_auto, (0, 0, 255))
    
    for i, (im, bbox) in enumerate(zip(imgs,gt)):
        bbox = map(int, bbox)
        cv2.rectangle(im, (bbox[0], bbox[1]), (
            bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 255, 0), 1)
        
        cv2.imshow('tracking', im)
        cv2.waitKey(10)
        
