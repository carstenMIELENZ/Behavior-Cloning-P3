#################################################################################
#
# model.pv 
# + contains CNN model creation, training and save
# + parameters used for submitted model.h5 
#   TakeNum = 10 
#   steering_correction = 0.25 
#   steering_threshold  = 0.21 
#   nepochs = 6
#
#################################################################################

import csv
import cv2
import numpy as np
import sklearn

#################################################################################
#
# Read CVS file (with comment line) + filtering straight images 
#
#################################################################################
lines = []

comment_blocker  = True         # ignore commment 1st line

# Filter
count   = 0
removed = 0  # straight images removed/filtered
number  = 0  # number of straight 
TakeNum = 10 # take every TakeNum images 

# open CVS file
with open ('./Data/driving_log.csv') as csvfile :
    reader = csv.reader(csvfile)
    for line in reader :
        
        # Comment blocker only needed for reference
        if comment_blocker :            
            comment_blocker = False
            continue
            
        # Check steering value for straight images
        if (float(line[3]) == 0) :
            # Straight Filter
            number = number + 1
            count = count + 1
            if (count ==  TakeNum) :  # add straight image
                lines.append(line)
                count = 0
            else :
                removed = removed + 1
        else :
            # add 
            lines.append(line)      # add non-straight image

print ()                    
print ('debug: total images',len(lines),'- straight images =',number,', removed straght images =',removed)
print ()                    

#################################################################################
#
# Create camera file (center, left, right)
#
#################################################################################
steering_correction = 0.25  # correction value for right, left 

# master list
name_list_cam       = []
steering_list_cam   = []

# detail lists
name_list_cam_c     = []
steering_list_cam_c = []
name_list_cam_l     = []
steering_list_cam_l = []
name_list_cam_r     = []
steering_list_cam_r = []

for x in lines :

	# Centre Camera
	name     = './Data/IMG/' + x[0].split('/')[-1]
	steering = float(x[3])
	name_list_cam_c.append(name)
	steering_list_cam_c.append(steering)
        # Left Camera
	name     = './Data/IMG/' + x[1].split('/')[-1]
	steering = float(x[3]) + steering_correction
	name_list_cam_l.append(name)
	steering_list_cam_l.append(steering)
	# Right Camera
	name     = './Data/IMG/' + x[2].split('/')[-1]
	steering = float(x[3]) - steering_correction
	name_list_cam_r.append(name)
	steering_list_cam_r.append(steering)

# build master list
name_list_cam     = name_list_cam_c + name_list_cam_l + name_list_cam_r 
steering_list_cam = steering_list_cam_c + steering_list_cam_l + steering_list_cam_r

print ()
print ('debug: Centre Camera list size',len(steering_list_cam_c))
print ('debug:    3 x Camera list size',len(steering_list_cam))

#################################################################################
#
# Create augmentation file
#
#################################################################################
name_list_aug = []
steering_list_aug = []

steering_threshold = 0.21   # threshold for generating augmented images

TYPE1 = 100.0               # steering offset to mark augmented images, offset is removed in generator


for i in range(len(name_list_cam)) :

        # Check threshold for augmentation 
        if (abs(steering_list_cam[i]) > steering_threshold) :
            name_list_aug.append(name_list_cam[i])
            steering_list_aug.append((steering_list_cam[i]*-1.0)+TYPE1) # TYPE1 mark augmented image
            
# New data
name_list_full     = name_list_cam + name_list_aug
steering_list_full = steering_list_cam + steering_list_aug


print ('debug: total list size',len(steering_list_full))

#################################################################################
#
# process66x200: pre-processes simulator images for NVIDIA CNN model
#
# input image:   160x320x3 BGR
# process:       crop top by 60, bottom by 20, resize to 66x200x3, color BGR2YUV
# output image:  66x200x3 YUV
#
#################################################################################
import cv2
import numpy as np
import sklearn

def process66x200(image) :
	# crop 60 from top, remove 20 from bottom
	image_p = image[70:140,:]
	# resize
	image_p = cv2.resize(image_p,(200,66 ))
	# color convert to YUV
	return cv2.cvtColor(image_p, cv2.COLOR_BGR2YUV)

#################################################################################
#
# Generator
#
#################################################################################
def generator(images, labels, batch_size=32):
    
    num_samples = len(images)
    
    while 1: # Loop forever so the generator never terminates
     
        sklearn.utils.shuffle(images, labels)
        
        for offset in range(0, num_samples, batch_size):
            
            x_batch = images[offset:offset+batch_size]
            y_batch = labels[offset:offset+batch_size]

            x_sample = []
            y_sample = []
            
            for x_, y_ in zip (x_batch, y_batch) :
                
                image = cv2.imread(x_)
                label = y_

                if (label > 50.0) :
                    label = (label - TYPE1) # remove augmentation marker
                    image = cv2.flip(image,1)
		
                image = process66x200(image)
                
                x_sample.append(image)
                y_sample.append(label)

            X_train = np.array(x_sample)
            y_train = np.array(y_sample)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
#################################################################################
#
# Split training & validation 
#
#################################################################################
from sklearn.model_selection import train_test_split

# Main list
X_train = np.array(name_list_full) 
y_train = np.array(steering_list_full)

# Get randomized datasets for training and validation
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=0.2,random_state=0)

# compile and train the model using the generator function
train_generator      = generator(X_train, y_train, batch_size=64)
validation_generator = generator(X_validation, y_validation, batch_size=64)

print ('debug: train, validation data set size',len(X_train),len(X_validation))

#################################################################################
#
# CNN Model - NVIDIA (66,200,3)
#
#################################################################################
print ('debug: checking Augmented Steering Angle element[0] =',steering_list_aug[0])
print ('debug: creating model ...')
print ()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Activation, Dropout
from keras.layers import Lambda, Cropping2D, Reshape
from keras import optimizers  


model = Sequential()

# Normalization Layer
model.add(Lambda(lambda x: (x / 127.5)-1.0,input_shape=(66,200,3)))
#model.add(Cropping2D(cropping=((70,20), (0,0))))

# Next 3 Convolutional Layers

model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid'))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid'))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid'))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

# Next 2 Convolutional Layers

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# 4 Fully Connected Layers

model.add(Flatten())

model.add(Dense(100))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

model.add(Dense(1))

# Compile
model.compile(loss='mse', optimizer='adam')

#################################################################################
#
# Train model
#
#################################################################################
print ()
print ('debug: training model...')
print ()

nepochs = 6 

history_object = model.fit_generator(train_generator, samples_per_epoch=len(X_train),validation_data=validation_generator, nb_val_samples=len(X_validation), nb_epoch=nepochs)

#################################################################################
#
# Save model
#
#################################################################################
model_run = "model-%i-%i_e%i_sc%.3f-st%.3f.h5" % (TakeNum,len(X_train),nepochs,steering_correction,steering_threshold)

print ()
print ('debug: saved model:',model_run)
print ()

model.save(model_run)
