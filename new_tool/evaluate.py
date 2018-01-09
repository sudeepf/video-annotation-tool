
def get_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])
    
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou


def evaluate_estimation_iou(est, gt):
    '''
    retuns the list of ious
    :param est:
    :param gt:
    :return:
    '''
    ious = [get_intersection_over_union(gt[i], est[i][0]) for i in range(len(
        est))]
    avg_annotations = 0
    for e in est:
        eo = e[1]
        if not eo:
            avg_annotations += 1
    
    avg_annotations = float(len(ious)) / float(avg_annotations)
    
    return ious, avg_annotations


def evaluate_accuracy(ious, robustness_th = 0.66):
    success_point = 0
    for iou in ious:
        if iou > robustness_th:
            success_point += 1.0
    rate = success_point / len(ious)
    
    return rate