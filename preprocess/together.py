import os
import pandas as pd
import numpy as np

def get_paths(path,idxx,exts):
    idxs=[idxx,idxx[-6:],idxx[-4:]]
    for idx in idxs:
        for ext in exts:
            paths=path+'/'+idx+ext
            if os.path.exists(paths):
                return paths
    return None

def items_del():
    pose_path='/hdd/sdd/lzq/dianwang/dataset/pose_data'
    items=[]
    with open('/hdd/sdd/lzq/dianwang/dataset/items.txt','r') as f:
        items=eval(f.read())
    no=[]
    for i in os.listdir(pose_path):
        if len(os.listdir(pose_path+'/'+i+'/one_pose'))==0:
            no.append(i)
    print(no)
    new_items=[]
    for item in items:
        frame=item[0][-1]
        if os.path.split(frame)[-1] not in no:
            new_items.append(item)
    with open('/hdd/sdd/lzq/dianwang/dataset/items.txt','w') as f:
        f.write(str(new_items))


def main():
    labels=pd.read_excel('/hdd/sdd/lzq/dianwang/dataset/labels.xlsx')

    items=[]
    
    pose_cnt=0
    va_cnt=0

    for i in range(718):
        item=[]
        idx=labels.iloc[i]['编号']
        pose_path='/hdd/sdd/lzq/dianwang/dataset/pose_data'
        video_path='/hdd/sdd/lzq/dianwang/dataset/face_data'   
        audio_path='/hdd/sdd/lzq/dianwang/dataset/voice_data/voice_data'
        pose_path=get_paths(pose_path,idx,['.MP4','.MTS'])

        if pose_path is not None and os.path.exists(pose_path+'/normal_pose') and len(os.listdir(pose_path+'/normal_pose'))>0:
            pose_cnt+=1
            video_path=get_paths(video_path,idx,['.MP4','.MTS'])
            if video_path is not None:
                va_cnt+=1
                audio_path=get_paths(audio_path,idx,['.WAV','.wav'])
                if audio_path is not None:
                    
                    audio_pngs=list(filter(lambda x:x.endswith('.png'),os.listdir(audio_path+'/pic')))
                    for video_png in list(filter(lambda x:x.endswith('.png'),os.listdir(video_path+'/pic'))):
                        if video_png in audio_pngs:
                            item.append([video_path+'/pic/'+video_png,audio_path+'/pic/'+video_png,pose_path])
        if len(item)>0:
            label=labels.iloc[i]['标签']
            item.append(label)
            items.append(item)
    
    print(pose_cnt,va_cnt)

    file=open('/hdd/sdd/lzq/dianwang/dataset/items.txt','w') 
    file.write(str(items))
    file.close() 
    print(len(items))


main()
#items_del()
# f = open("/hdd/sdd/lzq/dianwang/dataset/items.txt","r")
# items = eval(f.read())
# print(len(items))
# f.close()
