#import liblary
import eval_segm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


#function post processing (3 modes)
def post_processing(img, post_process):
    post_processing_method = 'No Postprocessing'
    if post_process == 1:
        _, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY) # Fix Threshold
        post_processing_method = 'Fix Threshold'
    elif post_process == 2:
        img = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15,2) # Gaussian Threshold
        post_processing_method = 'Gaussian Threshold'
    elif post_process == 3:
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu Threshold
        post_processing_method = 'Otsu Threshold'
    return img, post_processing_method

#choose post_processing mode
post_process = 1
post_processing_method = 'Otsu Threshold'
classes = 'plant'
path_true_label = "plant/test/label/*png"
path_predicted_label = "plant/test/test result/*.png"

#memasukan label dan predictlabel
#liblary glob digunakan untuk menggunkan format yang sama
true_label = glob.glob(path_true_label)
predicted_label = glob.glob(path_predicted_label)

# memasukan  gambar ke variabel true_label
true_label_img = []
for path in true_label:
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (256, 256))
    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    true_label_img.append(img)

# memasukan  gambar ke variabel redict_label   
predicted_label_img = []
for path in predicted_label:
    img = cv2.imread(path, 0)
    img, post_processing_method = post_processing(img,post_process)
    predicted_label_img.append(img)
    

#membuat array untuk menampung nilai
    
list_pixel_acc = []
list_mean_acc = []
list_mean_IU = []
list_fpr = []
list_f1_score = []
list_recall = []
list_precision = []

# menampung hasil di result format.csv
with open('{} result {}.csv'.format(classes, post_processing_method), 'w') as fsave:
    for idx in range(len(true_label_img)):
        
        pixel_acc = eval_segm.pixel_accuracy(predicted_label_img[idx], true_label_img[idx])
        list_pixel_acc.append(pixel_acc)
        mean_acc = eval_segm.mean_accuracy(predicted_label_img[idx], true_label_img[idx])
        list_mean_acc.append(mean_acc)
        mean_UI = eval_segm.mean_IU(predicted_label_img[idx], true_label_img[idx])
        list_mean_IU.append(mean_UI)
        pred_segm = predicted_label_img[idx].copy()
        gt_segm = true_label_img[idx].copy()
        fpr = eval_segm.get_fpr(pred_segm, gt_segm)
        precision, recall, f1 = eval_segm.get_all(pred_segm, gt_segm)
        list_f1_score.append(f1)
        list_precision.append(precision)
        list_recall.append(recall)
        list_fpr.append(fpr)
        
# =============================================================================
#         filename = true_label[idx].split('\\')[-1]
#         filename = filename.split('.')[0]
#         filename = filename.replace(',', ' ')
#         fsave.write('{},{},{},{}'.format(filename,pixel_acc,mean_UI,mean_acc))
#         fsave.write('\n')
# =============================================================================
    
# menghitung nilai evaluasi dan menampung dalam variabel
    
    
list_pixel_acc = np.array(list_pixel_acc)
list_mean_IU = np.array(list_mean_IU)
list_mean_acc = np.array(list_mean_acc)
list_fpr = np.array(list_fpr)
list_f1_score = np.array(list_f1_score)
list_precision = np.array(list_precision)
list_recall = np.array(list_recall)

# menampilkan nilai evaluasi

print(list_f1_score)

print("Pixel Acc : {0:.2f}".format(np.average(list_pixel_acc) * 100))
print("IoU : {0:.2f}".format(np.average(list_mean_IU) * 100))
print("Mean Acc : {0:.2f}".format(np.average(list_mean_acc) * 100))
print("FPR : {0:.2f}".format(np.average(list_fpr) * 100))
print("Precision : {0:.2f}".format(np.average(list_precision) * 100))
print("Recall : {0:.2f}".format(np.average(list_recall)  * 100))
print("F1 Score : {0:.2f}".format(np.average(list_f1_score)  * 100))
    
    
    
    
    