# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:48:54 2019

@author: ISYSRG.COM
"""

from model import *
from data import *
from matplotlib import pyplot as plt
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


epochs = 500
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


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(10,train_path,'image','label',data_gen_args,save_to_dir = None)
testGene = valGenerator(10,test_path,'image','label')

model = get_fractalunet()
#model = unet(model_name)

model_checkpoint = ModelCheckpoint(model_name, monitor='loss',verbose=1, save_best_only=True)
#model_checkpoint = ModelCheckpoint(pretrain_model_name, monitor='loss',verbose=1, save_best_only=True)

history_train = model.fit_generator(myGene,steps_per_epoch = batch_size, epochs=epochs,callbacks=[model_checkpoint], validation_steps=validation_steps,validation_data=testGene)




import glob
import os
test_path = glob.glob(test_path_image)
results_filename = os.listdir(test_path_label)

model = load_model('FractalNet_plant_epoch_temp_1000_batch_64')
testGene = testGenerator(test_path)
results = model.predict_generator(testGene, len(test_path),verbose=1)
saveResult(result_path, results_filename, results)





##






