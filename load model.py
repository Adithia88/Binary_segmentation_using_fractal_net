# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:48:54 2019

@author: ISYSRG.COM
"""

from model import *
from data import *
from matplotlib import pyplot as plt
import pickle
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


epochs = 1000
batch_size = 64
step_per_epochs = batch_size
validation_steps = batch_size


train_path = 'plant/train'
test_path = 'plant/test'
test_path_image = 'plant/test/image/*.png'
test_path_label = 'plant/test/label/'
result_path = 'plant/test/test result/'
model_name = 'FractalNet_plant_epoch_temp_{}_batch_{}.hdf5'.format(epochs,step_per_epochs)
history_name = 'FractalNet_plant_history.pickle'






import glob
import os
test_path = glob.glob(test_path_image)
results_filename = os.listdir(test_path_label)

model = load_model('FractalNet_plant_epoch_temp_500_batch_64.hdf5')
testGene = testGenerator(test_path)
results = model.predict_generator(testGene, len(test_path),verbose=1)
saveResult(result_path, results_filename, results)





##






