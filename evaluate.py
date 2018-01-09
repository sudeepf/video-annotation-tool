import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                else:
                    aDict = {element[0].tag: XmlListConfig(element)}
                if element.items():
                    aDict.update(dict(element.items()))
                if element.tag in self.keys():
                    self[element.tag].append(aDict)
                else:
                    self.update({element.tag: [aDict]})
            
            elif element.items():
                self.update({element.tag: dict(element.items())})
            else:
                self.update({element.tag: element.text})


def extract_rect(ddict):
    """
     This function returns a numpy array of Nx4 where N is number of frames
    currently supports only 1 object in the video
    :param ddict: input dict
    :return: numpy array
    """
    if len(ddict['object']) > 1:
        return NULL
    
    # list to store the data to return
    bbox_list = []
    # Format of data should be [top left h w]
    for frame in ddict['object'][0]['polygon']:
        bbox_list.append([int(frame['pt'][0]['x']), int(frame['pt'][0]['y']),
                          int(frame['pt'][3]['x']) - int(frame['pt'][1]['x']),
                          int(frame['pt'][1]['y']) - int(frame['pt'][0]['y']),])
    
    return np.array(bbox_list)
    
def extract_file(file_name):
    """
    This function extract bbox coordinates from the list in the given file
    :param file_name: file path of Ground truth
    :return: a numpy array of bounding boxes
    """
    return np.loadtxt(file_name, delimiter=',')


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


def get_accuracy_robustness_curve(ious, threshold_steps = 0.05):
    
    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump
    print ious
    acc = []
    rob = []
    for th in frange(0.0, 1.0, threshold_steps):
        count_tp = 0
        for iou in ious:
            if iou > th:
                count_tp += 1.0
        
        acc.append(count_tp / float(len(ious)))
        rob.append(th)
    
    return acc, rob

e = ET.parse('ValidationSet/Jump/output.xml')
root = e.getroot()
xmldict = XmlDictConfig(root)

# Now we extract the data from the dict and save it to the an array
estimation = extract_rect(xmldict)

# Get the groundtruth data
ground_truth = extract_file('ValidationSet/Jump/groundtruth_rect.txt')

# lets get IoU for all the bboxes
matching_score = [get_intersection_over_union(
    ground_truth[i,:], estimation[i,:]) for i in range(
    min(np.shape(ground_truth)[0], np.shape(estimation)[0]))]

# Get accuracy vs robustness curve
accuracy, robustness = get_accuracy_robustness_curve(matching_score)

# Now lets plot the iou score for each frames
plt.plot(matching_score)
plt.show()

# Now plot acc-bob curve
plt.plot(robustness, accuracy)
plt.show()

print np.mean(np.array(matching_score))
