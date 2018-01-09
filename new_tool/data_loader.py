from os import listdir
import numpy as np
from os.path import isfile, join
import cv2
from planar import BoundingBox

def extract_file(file_name):
    """
    This function extract bbox coordinates from the list in the given file
    :param file_name: file path of Ground truth
    :return: a numpy array of bounding boxes
    """
    boxes = np.loadtxt(file_name, delimiter=',')
    b_list = []
    for box in boxes:
        bbox = BoundingBox(list(box.reshape(-1,2)))
        b_list.append([list(bbox.min_point) + [bbox.width, bbox.height]])
    
    return np.array(b_list).reshape(-1,4)
    

class DataLoader:
    def __init__(self, folder_path):
        self.onlyFolders = [join(folder_path, f) for f in listdir(
            folder_path) if isfile(join(folder_path, f)) != 1]
        # Each folder is the damn data point
        self.onlyFolders.sort()
        self.count = 0
        self.total_vids = len(self.onlyFolders)
    
    def get_next(self):
        if self.count >= self.total_vids:
            return None, None
        self.count += 1
        folder = self.onlyFolders[self.count-1]
        onlyImgs = [join(folder, f) for f in listdir(folder) if
                    f.split('.')[1] == 'jpg']
        onlyImgs.sort()
        onlyImgs = [cv2.imread(f) for f in onlyImgs]
        # Get the groundtruth data
        ground_truth = extract_file(join(self.onlyFolders[self.count-1],
                                    'groundtruth.txt'))
        
        return onlyImgs, ground_truth