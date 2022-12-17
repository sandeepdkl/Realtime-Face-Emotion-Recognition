import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras
from keras import layers


# img_array = cv2.imread("D:\\4. Python Codes\\Emotion Recognition\\train\\0\\Training_3908.jpg")
# print(img_array.shape) #rgb 
# plt.imshow(img_array) #bgr
# plt.show()

data_directory = "D:\\4. Python Codes\\Emotion Recognition\\train\\" #training dataset
classes = ["0", "1", "2", "3", "4", "5", "6"] #list of classes. Exact name of folders

img_size = 224 #ImageNet => 224 x 224
training_data = []

def create_training_data():
    for category in classes:
        path = os.path.join(data_directory, category)
        class_num = classes.index(category) #0 1 2 label
        for img in os.listdir(path): #returns iterable containing image file names which can be a list of strings. 
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array, (img_size, img_size) )
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()


# plt.imshow(cv2.cvtColor(training_data[10001][0], cv2.COLOR_BGR2RGB))
# plt.show()  


random.shuffle(training_data)

X = []  #data or feature
Y = []  #label

for features, label in training_data:
    X.append(features)
    Y.append(label)
    
X = np.array(X).reshape(-1, img_size, img_size, 3)  #converting into 4 dimension. Mobilenet takes 4 dimension

# print(X.shape)

Y = np.array(Y)

model = tf.keras.applications.MobileNetV2() #pretrained model

# print(model.summary())

base_input = model.layers[0].input #taking the first layer
base_output = model.layers[-2].output #taking only upto the second last layer
final_output = layers.Dense(128)(base_output)  #adding a new layer after the output of global pooling layer
final_output = layers.Activation('relu')(final_output) #activation function
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation = 'softmax')(final_output) #classification layer
new_model = keras.Model(inputs = base_input, outputs = final_output)
# print(new_model.summary())

new_model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

new_model = tf.keras.models.load_model('D:\\4. Python Codes\\Emotion Recognition\\Final_model_95p07.h5')
frame = cv2.imread("D:\\4. Python Codes\\Emotion Recognition\\happy_child.jpg")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# plt.show()

faces = faceCascade.detectMultiScale(gray, 1.3,5)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) #BGR
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for(ex,ey,ew,eh) in facess:
            face_roi = roi_color[ey: ey+eh, ex: ex+ew]

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.show()

# plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
# plt.show()
final_image = cv2.resize(face_roi, (224,224))  ##224 x 224 resizing
final_image = np.expand_dims(final_image, axis = 0) #need fourth dimension

Predictions = new_model.predict(final_image)

print(np.argmax(Predictions))  #3 is happy

            



cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    

    faces = faceCascade.detectMultiScale(gray, 1.3,5)
      

    # Draw a rectangle around the faces
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) #BGR
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex: ex+ew]
    final_image = cv2.resize(face_roi, (224,224))  ##224 x 224 resizing
    final_image = np.expand_dims(final_image, axis = 0) #need fourth dimension
    
    font = cv2.FONT_HERSHEY_COMPLEX
    Predictions = new_model.predict(final_image)
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    
    if(np.argmax(Predictions) ==0):
       status = "Angry"
       x1,y1,w1,h1 = 0,0,175,75 #black rectangle
       cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
       cv2.putText(frame, status, (x1+ int(w1/10), y1+ int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255))
    elif(np.argmax(Predictions) ==1):
       status = "Disgust"
       x1,y1,w1,h1 = 0,0,175,75 #black rectangle
       cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
       cv2.putText(frame, status, (x1+ int(w1/10), y1+ int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255))
       
    elif(np.argmax(Predictions) ==2):
       status = "Fear"
       x1,y1,w1,h1 = 0,0,175,75 #black rectangle
       cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
       cv2.putText(frame, status, (x1+ int(w1/10), y1+ int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255))
       
    elif(np.argmax(Predictions) ==3):
       status = "Happy"
       x1,y1,w1,h1 = 0,0,175,75 #black rectangle
       cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
       cv2.putText(frame, status, (x1+ int(w1/10), y1+ int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255))
       
    elif(np.argmax(Predictions) ==4):
       status = "Neutral"
       x1,y1,w1,h1 = 0,0,175,75 #black rectangle
       cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
       cv2.putText(frame, status, (x1+ int(w1/10), y1+ int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255))
         
    elif(np.argmax(Predictions) ==5):
       status = "Sad"
       x1,y1,w1,h1 = 0,0,175,75 #black rectangle
       cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
       cv2.putText(frame, status, (x1+ int(w1/10), y1+ int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255))
    else:
       status = "Surprise"
       x1,y1,w1,h1 = 0,0,175,75 #black rectangle
       cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
       cv2.putText(frame, status, (x1+ int(w1/10), y1+ int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255))
      
    cv2.imshow('Face Emotion Recognition', frame)
    
    if(cv2.waitKey(2) & 0xFF == ord('q')):
         break
    
cap.release()
cv2.destroyAllWindows()
          
