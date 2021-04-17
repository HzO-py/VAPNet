import os
import json
from math import sqrt,atan2

pi=3.1415906535
path="/hdd/sdd/lzq/dianwang/dataset/pose_data"

def dis(p1,p2):
    if p1==p2:
        return 0 
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def angle(p1,p2):
    if p1==p2:
        return 0 
    else:
        return atan2((p1[1]-p2[1]),(p1[0]-p2[0]))/pi

def choose_pose(m_pose,poses):
    ans_pose=m_pose
    min_dis=1e9
    
    for pose in poses:
        pose=pose["pose_keypoints_2d"]

        dis=0
        point_num=0

        for i in range(0,25*3,3):
            if pose[i]!=0 and pose[i+1]!=0 and pose[i+2]!=0:
                dis+=(pose[i]-m_pose[i])**2+(pose[i+1]-m_pose[i+1])**2
                point_num+=1

        if point_num==0:
            continue

        dis/=point_num
        if dis<min_dis:
            min_dis=dis
            ans_pose=pose

    for i in range(0,25*3,3):
        if ans_pose[i]==0 or ans_pose[i+1]==0 or ans_pose[i+2]==0:
            ans_pose[i]=m_pose[i]
            ans_pose[i+1]=m_pose[i+1]
            ans_pose[i+2]=m_pose[i+2]

    return ans_pose

def one_person():
    no=[]

    for f_name in sorted(os.listdir(path)):
        try:
            if not os.path.exists(path+'/'+f_name+'/one_pose'):
                os.makedirs(path+'/'+f_name+'/one_pose')

            i=0
            m_pose=[]
            ok=False

            poses=sorted(list(filter(lambda x:x.endswith('.json'),os.listdir(path+'/'+f_name))))

            for pose in poses:
                with open(path+'/'+f_name+'/'+pose) as f:
                    js=json.load(f)

                    if len(js["people"])==1 and 0 not in js["people"][0]["pose_keypoints_2d"]:
                        m_pose=js["people"][0]["pose_keypoints_2d"]
                        ok=True
                        with open(path+'/'+f_name+'/one_pose/'+os.path.splitext(pose)[0]+'.txt','w') as ff:
                            ff.write(str(m_pose))
                        break
                i+=1

            if ok==False:
                no.append(f_name)
                continue

            n_pose=m_pose
            for j in range(i-1,-1,-1):
                with open(path+'/'+f_name+'/'+poses[j]) as f:
                    js=json.load(f)
                    ans_pose=choose_pose(m_pose,js["people"])
                    m_pose=ans_pose
                    with open(path+'/'+f_name+'/one_pose/'+os.path.splitext(poses[j])[0]+'.txt','w') as ff:
                        ff.write(str(ans_pose))

            m_pose=n_pose
            for j in range(i+1,len(poses)):
                with open(path+'/'+f_name+'/'+poses[j]) as f:
                    js=json.load(f)
                    ans_pose=choose_pose(m_pose,js["people"])
                    m_pose=ans_pose
                    with open(path+'/'+f_name+'/one_pose/'+os.path.splitext(poses[j])[0]+'.txt','w') as ff:
                        ff.write(str(ans_pose))

            print(f_name)

        except:
            no.append(f_name)

    print(no)


def normal_pose():
    up_points=[0,1,2,3,4,5,6,7,15,16,17,18]
    for f_name in sorted(os.listdir(path)):
        for p_name in sorted(os.listdir(path+'/'+f_name+'/one_pose')):
            with open(path+'/'+f_name+'/one_pose/'+p_name,'r') as f:
                pose=eval(f.read())
                new_pose=[]
                dis_nh=dis((pose[1*3+0],pose[1*3+1]),(pose[8*3+0],pose[8*3+1]))
                for i in range(0,25*3,3):
                    if i//3 in up_points:
                        center=1
                    else:
                        center=8
                    new_point_dis=dis((pose[i],pose[i+1]),(pose[center*3+0],pose[center*3+1]))/dis_nh
                    new_point_angle=angle((pose[i],pose[i+1]),(pose[center*3+0],pose[center*3+1]))
                    new_pose.append([new_point_dis,new_point_angle])
                if not os.path.exists(path+'/'+f_name+'/normal_pose'):
                    os.makedirs(path+'/'+f_name+'/normal_pose')
                with open(path+'/'+f_name+'/normal_pose/'+os.path.splitext(p_name)[0]+'.txt','w') as ff:
                    ff.write(str(new_pose))
        print(f_name)

#print(atan(-2/(0.0001)))
normal_pose()