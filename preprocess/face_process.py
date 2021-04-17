import os
import cv2
import numpy as np
import random
import shutil

model = cv2.CascadeClassifier('/hdd/sdd/lzq/dianwang/model/haarcascade_frontalface_alt2.xml')
dataset_path='/hdd/sdc/hxj_dataset/电网数据'
save_path='/hdd/sdd/lzq/dianwang/dataset/face_data'
problem=[]

def listmp4(subpath):
    for subpath_file in os.listdir(subpath):
        ab_path=subpath+'/'+subpath_file
        if os.path.isdir(ab_path):
            listmp4(ab_path)
        elif os.path.isfile(ab_path):
            if '演讲' in ab_path and '步态' not in ab_path:
                file_name,file_ext=os.path.splitext(subpath_file)
                if file_ext=='.MP4' or file_ext=='.MTS':
                    #print(ab_path)
                    try:
                        video2img(ab_path)
                    except:
                        problem.append(ab_path)

def video2img(file):
    cap = cv2.VideoCapture(file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    flag,frame=cap.read()
    i=0
    j=0
    rd=random.randint(0,fps*3-1)
    save_filedir_path=save_path+'/'+os.path.split(file)[-1]+'/pic'
    if not os.path.exists(save_filedir_path):
        os.makedirs(save_filedir_path)
    if len(list(filter(lambda x:x.endswith('.png'),os.listdir(save_filedir_path))))>=10:
        return
    while flag:
        if j==fps*3:
            j=0
            i+=1
            rd=random.randint(0,fps*3-1)
        if j==rd:
            h, w=frame.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY),-90,1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY       
            frame=cv2.warpAffine(frame,M,(nW,nH))
            face_img=cut_face(frame)
            if face_img is not None:
                cv2.imwrite(save_filedir_path+'/'+str(i)+'.png',face_img)
                print(save_filedir_path+'/'+str(i)+'.png')
        flag,frame=cap.read()
        j+=1

def cut_face(img):
    #gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    faces = model.detectMultiScale(img,scaleFactor=1.1,minNeighbors=7,minSize=(100,100))
    if len(faces)==0 or len(faces)>1:
        return None
    (x,y,w,h)=faces[0]
    face_img=img[y:y+h,x:x+w]
    face_img=cv2.resize(face_img,(128,128))
    return face_img

# def pic_mkdir(p):
#     for f in os.listdir(p):
#         if not os.path.exists(p+'/'+f+'/pic'):
#             os.makedirs(p+'/'+f+'/pic')
#         else:
#             shutil.rmtree(p+'/'+f+'/pic')
#         for png in os.listdir(p+'/'+f):
#             if png.endswith('.png'):
#                 shutil.move(p+'/'+f+'/'+png, p+'/'+f+'/pic/'+png)
#         print(f)


listmp4(dataset_path)
#video2img('/hdd/sdc/hxj_dataset/电网数据/广州、中山（第一批）/广东电网调度中心/演讲/演讲视频/000087.MTS')
