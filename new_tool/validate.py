import data_loader
import tracking_methods as methods
import visualize
import evaluate
import numpy as np
import pickle

validation_folder = 'ValidationSet'

max_th = 50
min_th = 10
inter_th = 10

accuracy_th = 0.5

def main():
    
    current_th = min_th
    rate_list_auto = []
    interval_list = []
    rate_list_lin = []
    
    while current_th < max_th:
    
        loader = data_loader.DataLoader(validation_folder)
        all_ious = []
        all_itervals = []
        while (True):
            imgs, gts = loader.get_next()
        
            if imgs == None:
                break
            # Do the auto tracking
            pred_auto = methods.auto_select(imgs, gts, stride=current_th)
            iou, est_interval = evaluate.evaluate_estimation_iou(pred_auto, gts)
            # evaluate the system
            all_ious += iou
            all_itervals.append(est_interval)
            
            rate_list_auto.append(evaluate.evaluate_accuracy(iou, accuracy_th))

            interval_list.append(1./est_interval)

            pred_lin = methods.linear_annotation(imgs, gts,
                                                 stride=current_th)
            iou, est_interval = evaluate.evaluate_estimation_iou(pred_lin, gts)

            rate_list_lin.append(evaluate.evaluate_accuracy(iou, accuracy_th))
            
            print("Processed data point - ", len(rate_list_auto))
            
            visualize.visualize_video(imgs, pred_lin, pred_auto, gts)
        current_th += inter_th
    
        print ("Evaluating for TH = ", current_th)
    
    pickle.dump([rate_list_lin, rate_list_auto, interval_list],
                open("save2f.p", "wb"))
    
    
if __name__ == '__main__':
    main()