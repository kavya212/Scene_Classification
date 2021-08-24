#1
from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import cv2
import os

#2
img_height, img_width = (224,224)
batch_size = 32

train_data_dir = r"train"
valid_data_dir = r"val"
test_data_dir = r"test"

#4
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.4)

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')#set as training data

valid_generator=train_datagen.flow_from_directory(
    valid_data_dir, #same dir as training data
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')# set as validation data

'''#add
class_names = train_datagen.class_names
print(class_names)'''

#5
test_generator=train_datagen.flow_from_directory(
    test_data_dir, #same dir as training data
    target_size=(img_height,img_width),
    batch_size=1,
    class_mode='categorical',
    subset='validation')# set as validation data

#add
print(valid_generator.class_indices)
#6
x,y=test_generator.next()
x.shape

#11
base_model=ResNet50(include_top=False,weights='imagenet')
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
predictions = Dense(train_generator.num_classes,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
    layer.trainable=False

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_generator,epochs=10)

#12
model.save('Saved_Model_final.h5')

#13
test_loss, test_acc=model.evaluate(test_generator,verbose=2)
print('\nTest accuracy:',test_acc)

