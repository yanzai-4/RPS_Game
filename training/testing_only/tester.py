
import matplotlib
matplotlib.use('Agg')
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import sys

#加载模型h5文件
model = load_model("C:\D\Download\RPS-131\\new\\rock_paper_scissors_cnn.h5")
#model.summary()
#规范化图片大小和像素值

def get_inputs(file_path):
    img = image.load_img(file_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor=np.expand_dims(img_tensor,axis=0)
    img_tensor/=255.
    return img_tensor

#调用函数，规范化图片

#预测
cap = cv2.VideoCapture(0)

SAVE_PATH = os.getcwd()+'\\temp'

try:
    os.mkdir(SAVE_PATH)
except FileExistsError:
    pass

ct = 0
maxCt = 20
temp = []
print("Hit Space to Capture Image")

while True:
    ret, frame = cap.read()
    cv2.imshow('Get Data : ',frame[50:350,150:450])
    if cv2.waitKey(1) & 0xFF == ord(' '):
        img_PATH=SAVE_PATH+'\\'+'{}.jpg'.format(ct)
        cv2.imwrite(img_PATH,frame[50:350,150:450])
        pre_x = get_inputs(img_PATH)
        pre_y = model.predict(pre_x)
        print (pre_y)
        e = np.argmax(pre_y)
        if(e==0):
            print("Rock")
        elif(e==1):
            print("paper")
        elif(e==2):
            print("Scissor")
        else:
            print("Not recognized")
        #print(SAVE_PATH+'\\'+label+'_'+'{}.jpg Captured'.format(ct))
        ct+=1
    if ct >= maxCt:
        break

cap.release()
cv2.destroyAllWindows()




