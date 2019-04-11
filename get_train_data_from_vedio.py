# coding=utf-8
import os
import cv2
import argparse
import numpy as np
import Auto_train
import time
import glob
from AI_train import get_gray_img

H = 84
W = 84
C = 4
TIME_GAP = 0.1
TARGET_WIDTH = 1920//2
TARGET_HEIGHT = 1080//2
SKIP_NUM = 100

def get_all_npy_from_tfs(tfs_path):
    dir_list = Auto_train.get_dir(tfs_path)
    whole_data = []
    whole_label = []
    for dir in dir_list:
        npy_path = tfs_path+'\\'+dir
        files = os.listdir(npy_path)
        npy_name_list = list(filter(lambda y: '.npy' in y, files))
        print('开始合并%s目录下面的npy' % dir)
        for npy_name in npy_name_list:
            if '_data_' in npy_name:
                #print("train data %s" % npy_name)
                data_list = np.load(os.path.join(npy_path, npy_name))
                whole_data.append(data_list)
            elif '_label_' in npy_name:
                #print("train label %s" % npy_name)
                label_list = np.load(os.path.join(npy_path, npy_name))
                whole_label.append(label_list)
            else:
                print('没有npy文件需要合并')

    whole_data = np.concatenate(whole_data, axis=0)
    whole_label = np.concatenate(whole_label, axis=0)

    print(whole_data.shape)
    print(whole_label.shape)

    np.save('whole_data_front_all.npy', whole_data)
    np.save('whole_label_all.npy', whole_label)
    #tfs备份all文件
    np.save(tfs_path+'\\'+'whole_data_front_all.npy', whole_data)
    np.save(tfs_path+'\\'+'whole_label_all.npy', whole_label)

def read_vedio_from_tfs(tfs_path):
    vedio_dir_list  =Auto_train.get_train_video(tfs_path)
    print("当前需要训练的赛道list：")
    print(vedio_dir_list)
    for vedio_dir in vedio_dir_list:
        vedio_path = tfs_path+'\\'+vedio_dir
        print('获取训练数据%s'%vedio_path)
        files = os.listdir(vedio_path)
        vedio_name_list = list(filter(lambda x: '.mp4' in x, files))
        npy_name_list = list(filter(lambda y: '.npy' in y, files))
        for vedio_name in vedio_name_list:
            if 'train_data_'+vedio_name[:-4]+'.npy' in npy_name_list:
                print("已经存在视频%s的npy文件，不需要解析"%vedio_name)
            else:
                print("解析视频文件%s"%vedio_name)
                read_vedio_kernel(vedio_path, vedio_name)

        print("%s的录像文件采集完毕" % vedio_dir)
        Auto_train.write_vedio_to_file(vedio_dir, tfs_path)

        '''
        whole_data = []
        whole_label = []

        for vedio_name in vedio_name_list:
            data_list = np.load(os.path.join(vedio_path, 'train_data_' + vedio_name[:-4] + '.npy'))
            label_list = np.load(os.path.join(vedio_path, 'train_label_' + vedio_name[:-4] + '.npy'))

            whole_data.append(data_list)
            whole_label.append(label_list)

        whole_data = np.concatenate(whole_data, axis=0)
        whole_label = np.concatenate(whole_label, axis=0)

        np.save('whole_data_front_' + vedio_dir + '.npy', whole_data)
        np.save('whole_label_' + vedio_dir + '.npy', whole_label)
       '''



def read_vedio(vedio_path):
    files = os.listdir(vedio_path)
    vedio_name_list = list(filter(lambda x: '.mp4' in x, files))
    npy_name_list = list(filter(lambda y: '.npy' in y, files))
    for vedio_name in vedio_name_list:
        if 'train_data_' + vedio_name[:-4] + '.npy' in npy_name_list:
            print("已经存在视频%s的npy文件，不需要解析" % vedio_name)
        else:
            print("解析视频文件%s" % vedio_name)
            read_vedio_kernel(vedio_path, vedio_name)

    #for vedio_name in vedio_name_list:
        #read_vedio_kernel(vedio_path, vedio_name)

    whole_data = []
    whole_label = []

    print('vedio_name_list', vedio_name_list)

    for vedio_name in vedio_name_list:
        data_list = np.load(os.path.join(vedio_path, 'train_data_'+vedio_name[:-4]+'.npy'))
        label_list = np.load(os.path.join(vedio_path, 'train_label_'+vedio_name[:-4]+'.npy'))

        whole_data.append(data_list)
        whole_label.append(label_list)


    whole_data = np.concatenate(whole_data, axis=0)
    whole_label = np.concatenate(whole_label, axis=0)

    np.save('whole_data_front_'+vedio_path+'.npy', whole_data)
    np.save('whole_label_'+vedio_path+'.npy', whole_label)


def read_vedio_kernel(vedio_path, vedio_name):
    videoCapture = cv2.VideoCapture(os.path.join(vedio_path, vedio_name))
      
    #获得码率及尺寸
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(fps, size)
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
     

    def get_gray_img(img):
        #31~35
        sub_img = img[30:180, 800: 940]
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        sub_img = cv2.resize(sub_img, (H, W))
        return sub_img


    def get_gray_img1(img):
        sub_img = img[65:170, 805: 940]  #minicap
        #sub_img = img[92:190, 810: 940]   #摄像头
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        sub_img = cv2.resize(sub_img, (H, W))
        #sub_img = self.get_feature_img(sub_img)
        sub_img = ((sub_img > 150) * 255).astype(np.uint8)
        return sub_img

    def get_left_idx(full_img):
        #16~23
        height_start, height_end = 760//2, 840//2
        width_start, width_end = 175//2, 240//2

        temp_sub_img = full_img[height_start:height_end, width_start:width_end, :]
        cv2.imshow('left', temp_sub_img)
        temp_sub_img_HSV = cv2.cvtColor(temp_sub_img, cv2.COLOR_BGR2HSV)
        mean_H = np.average(temp_sub_img_HSV[:,:,-1])
        if mean_H > 230:
            return True, mean_H
        else:
            return False, mean_H

    def get_right_idx(full_img):
        #16~23
        height_start, height_end = 760//2, 840//2
        width_start, width_end = 475//2, 540//2

        temp_sub_img = full_img[height_start:height_end, width_start:width_end, :]
        cv2.imshow('right', temp_sub_img)
        temp_sub_img_HSV = cv2.cvtColor(temp_sub_img, cv2.COLOR_BGR2HSV)
        mean_H = np.average(temp_sub_img_HSV[:,:,-1])
        if mean_H > 230:
            return True, mean_H
        else:
            return False, mean_H

    def get_drift_idx(full_img):
        #1~5
        height_start, height_end = 713//2, 908//2
        width_start, width_end = 1532//2, 1712//2
        temp_sub_img = full_img[height_start:height_end, width_start:width_end, :]
        cv2.imshow('drift', temp_sub_img)
        temp_sub_img_HSV = cv2.cvtColor(temp_sub_img, cv2.COLOR_BGR2HSV)
        mask = np.zeros((height_end-height_start, width_end-width_start,3)).astype(np.uint8)
        cv2.circle(mask,(100//2,100//2),80//2,(255,255,255),-1)
        mask = (mask[:,:,0]/255).astype(np.uint8)
        H = temp_sub_img_HSV[:,:,-1] * mask
        #cv2.imshow('drift2', H)
        mean_H = np.mean(np.mean(H))
        if mean_H > 120:
            return True, mean_H
        else:
            return False, mean_H

    success, frame = videoCapture.read()
    step = 0
    current_frame = 0
    left_list = []
    right_list = []
    drift_list = []

    image_list = []
    label_list = []
    while success:
        pic = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

        #模板匹配
        #if template_find_pic('shache.png',pic,t=0.5) is False:
            #success, frame = videoCapture.read()  # 获取下一帧
            #continue

        mini_map = get_gray_img(pic)
        right_pressed, r_H = get_right_idx(pic)
        left_pressed, l_H = get_left_idx(pic)
        shift_pressed, d_H = get_drift_idx(pic)
        print(right_pressed, '%.3f'%r_H, left_pressed, '%.3f'%l_H, shift_pressed, '%.3f'%d_H)
        cv2.imshow('windows', mini_map) #显示
        #cv2.waitKey(1000//int(fps)) #延迟
        
        #Process
        current_time = current_frame * 1.0/fps
        if current_time < step*TIME_GAP:
            pass
            #left_list.append(left_pressed)
            #right_list.append(right_pressed)
            #drift_list.append(shift_pressed)
        else:
            step += 1
            image_list.append(mini_map)
            if left_pressed and shift_pressed:
                #左飘
                label_list.append(np.array([0, 0, 1, 0, 0]))
            elif left_pressed:
                #左转
                label_list.append(np.array([0, 1, 0, 0, 0]))
            elif right_pressed and shift_pressed:
                #右飘
                label_list.append(np.array([0, 0, 0, 0, 1]))
            elif right_pressed:
                #右转
                label_list.append(np.array([0, 0, 0, 1, 0]))
            else:
                #无动作
                label_list.append(np.array([1, 0, 0, 0, 0]))

            print("APPEND:", label_list[-1])

        cv2.waitKey(1)

        success, frame = videoCapture.read() #获取下一帧
        current_frame+=1

    videoCapture.release()


    print(len(image_list), len(label_list))
    data_list = np.array(image_list[:-1]).astype(np.uint8)
    label_list = np.array(label_list[1:])
    print(data_list.shape, label_list.shape)
    
    new_train_data = []
    new_train_label = []
    for i in range(3, len(data_list)):
        input_one_record = np.dstack(
            (
                data_list[i-3].ravel(),
                data_list[i-2].ravel(),
                data_list[i-1].ravel(),
                data_list[i].ravel()
            )
            ).ravel().reshape(H, W, C)
        new_train_data.append(input_one_record)
        new_train_label.append(label_list[i])
        if label_list[i][0] == 0:
            new_train_data.append(input_one_record)
            new_train_label.append(label_list[i])

    new_train_data = np.array(new_train_data)
    new_train_label = np.array(new_train_label)
    np.save(os.path.join(vedio_path, 'train_data_'+vedio_name[:-4]+'.npy'), new_train_data)
    np.save(os.path.join(vedio_path, 'train_label_'+vedio_name[:-4]+'.npy'), new_train_label)
    print('-------------------')
    print(new_train_data.shape)
    print(new_train_label.shape)

#测试代码
def template_find_pic(tpl_path, image, t=0.7):
    tpl_s = cv2.imread(tpl_path)  # 模板
    tpl = cv2.cvtColor(tpl_s, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #th, tw = tpl.shape[:2]
    result = cv2.matchTemplate(target, tpl, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(max_val)
    if max_val < t:
        print("没有找到刹车按钮，丢弃:")
        return False
    else:
        print("找到刹车按钮，保留")
        return True

def main():
    parser = argparse.ArgumentParser(description='PPGAME supervised process train data')
    parser.add_argument('--type', default='', help='Train or Test')
    parser.add_argument('--track', default=' ', help='Track Name')
    args = parser.parse_args()
    track_name = args.track

    if args.type == 'tfs':
        tfs_path = r'\\tencent.com\tfs\跨部门项目\光子开发资源库\AI训练源数据\PPGame'
        ready_flag = tfs_path+"\\"+"get_reay_flag.txt"
        print("从tfs自动训练")
        with open(ready_flag,'r') as f:
            flag = int(f.readline())
        if flag ==1:
            print("文件传输完毕，开始分析录像")
            read_vedio_from_tfs(tfs_path)
            with open(ready_flag,'w') as w:
                w.write('0')
            if os.path.exists(tfs_path+'\\'+'whole_data_front_all.npy'):#旧的npy文件进行备份
                os.rename(tfs_path+'\\'+'whole_data_front_all.npy',tfs_path+'\\'+'whole_data_front_all_'+time.strftime("%m%d%H%M%S",time.localtime())+'.npy')
                os.rename(tfs_path + '\\' + 'whole_label_all.npy',tfs_path + '\\' + 'whole_label_all_' + time.strftime("%m%d%H%M%S",time.localtime()) + '.npy')
            print('开始制作xx_all.npy文件')
            get_all_npy_from_tfs(tfs_path)
        else:
            print('请确认文件是否传输完毕，或者是否将文件传输标识写为1')
    else:
        read_vedio(track_name)


if __name__ == '__main__':
    main()
    #get_all_npy_from_tfs(r'E:\PythonProject\PPGAME_supervised_whiff\test_data')
    #read_vedio('sd20')




#np.save('whole_data_front_'+name+'.npy', new_data_list)
#np.save('whole_label_'+name+'.npy', new_label_list)