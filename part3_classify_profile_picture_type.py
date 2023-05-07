import glob
from sklearn.model_selection import KFold

import os
import shutil

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

import pickle
from tensorflow.keras.optimizers.legacy import Adam


if not os.path.exists('exam_data/part_3/training_data/test_images'):
  os.mkdir('exam_data/part_3/training_data/test_images/')

class_list = ['buildings', 'dogs', 'faces']
class_list_length  = len(class_list)

for i in class_list:
  if not os.path.exists('exam_data/part_3/training_data/test_images/' + i):
    os.mkdir('exam_data/part_3/training_data/test_images/' + i)


file_list = []
for i in class_list:
  file_list.append([i for i in glob.glob('exam_data/part_3/training_data/image_dataset/' + i + '/*.jpg')])
  
  
split_number = 4
kfold_object = KFold(n_splits=split_number)

index_list =  []
for i in range(class_list_length):
  kfold_object.get_n_splits(file_list[i])
  index_list.append([[training_index, test_index] for training_index, test_index in kfold_object.split(file_list[i])])
  

# for count in range(split_number):
#   print("round: ", count)
#   for i in range(class_list_length):
#     for j in index_list[i][count][1]:
#       # print(file_list[i][j], 'dataset/test_images/' + class_list[i] + '/' + file_list[i][j].split('/')[-1])
#       shutil.move(file_list[i][j], 'exam_data/part_3/training_data/test_images/' + class_list[i] + '/' + file_list[i][j].split('/')[-1])


#   train_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='exam_data/part_3/training_data/image_dataset/', target_size=(30,30), shuffle=True)
#   test_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='exam_data/part_3/training_data/test_images/', target_size=(30,30), shuffle=True)                 

#   values = list(train_dataset.class_indices.values())
#   keys = list(train_dataset.class_indices.keys())
#   print([[values[i], keys[i]] for i in range(len(values))])

#   machine = Sequential()
#   machine.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(30,30,3)))
#   machine.add(Activation('relu'))
#   machine.add(Conv2D(filters=32, kernel_size=(3,3)))
#   machine.add(Dropout(0.3))
#   machine.add(Activation('relu'))
#   machine.add(MaxPooling2D(pool_size=(2,2)))

#   machine.add(Conv2D(filters=64, kernel_size=(3,3)))
#   machine.add(Activation('relu'))
#   # machine.add(Conv2D(filters=128, kernel_size=(3,3)))
#   # machine.add(Activation('relu'))
#   machine.add(Conv2D(filters=64, kernel_size=(3,3)))
#   machine.add(Dropout(0.3))
#   machine.add(Activation('relu'))
#   machine.add(MaxPooling2D(pool_size=(2,2)))

#   machine.add(Flatten())
#   # machine.add(Dense(units=128, activation='relu', input_shape=(30,30,3)))
#   # machine.add(Dense(units=128, activation='relu'))
#   machine.add(Dense(units=64, activation='relu'))
#   machine.add(Dense(units=64, activation='relu'))
#   machine.add(Dropout(0.3))
#   machine.add(Dense(3, activation='softmax'))

#   # machine.summary()

#   machine.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

#   machine.fit(train_dataset, batch_size=128, epochs=30, validation_data=test_dataset) 


#   for i in range(class_list_length):
#     for j in class_list:
#       for file_name in os.listdir('exam_data/part_3/training_data/test_images/'+ j):
#           shutil.move(os.path.join('exam_data/part_3/training_data/test_images/' + j, file_name), 'exam_data/part_3/training_data/image_dataset/' + j)



train_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='exam_data/part_3/training_data/image_dataset/',
                                         target_size=(30,30), shuffle=True)
                                         
values = list(train_dataset.class_indices.values())
keys = list(train_dataset.class_indices.keys())
print([[values[i], keys[i]] for i in range(len(values))])


# Initializing the model
machine = Sequential()
machine.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(30,30,3)))
machine.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
machine.add(MaxPooling2D(pool_size=(2,2)))
machine.add(Dropout(0.3))

machine.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(MaxPooling2D(pool_size=(2,2)))
machine.add(Dropout(0.3))
          
machine.add(Flatten())
machine.add(Dense(units=64, activation='relu'))
machine.add(Dense(units=64, activation='relu'))
machine.add(Dropout(0.3))
machine.add(Dense(3, activation='softmax'))

# machine.summary()

machine.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
machine.fit(train_dataset, batch_size=128, epochs=30) 

pickle.dump(machine, open('machine_part3_cnn_image.pickle', 'wb'))