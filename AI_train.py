import os
import NN
import cv2
import  Auto_train
import shutil
import time
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from get_image import Get_Image
from controller import Controller
from test_reverse import test_reverse


H = 84
W = 84
C = 4
OUTPUT_SIZE = 5
BATCH_SIZE = 256
EPOCH_NUM = 40
LEARNING_RATE = 0.0005
MAX_STEPS = 1000
TIME_GAP = 0.1

#测试集和训练集分割函数.percent%的数据作为验证集
def my_train_test_split(whole_data_path,whole_label_path,percent):
    whole_data = np.load(whole_data_path)
    whole_label = np.load(whole_label_path)
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    print(whole_data.shape[0])
    random_index_array = np.random.randint(0, whole_data.shape[0], size=whole_data.shape[0])
    for index, x in enumerate( random_index_array):
        if (index + 1) % percent == 0:
            test_data.append(whole_data[random_index_array[index]])
            test_label.append(whole_label[random_index_array[index]])
        else:
            train_data.append(whole_data[random_index_array[index]])
            train_label.append(whole_label[random_index_array[index]])

    print('分割完毕')
    return train_data,test_data,train_label,test_label

#小喷操作
def get_whiff_idx(full_img):
    height_start, height_end = 694 // 2, 726 // 2
    width_start, width_end = 1414 // 2, 1448 // 2
    temp_sub_img = full_img[height_start:height_end, width_start:width_end, :]
    cv2.imshow('whiff', temp_sub_img)
    temp_sub_img_HSV = cv2.cvtColor(temp_sub_img, cv2.COLOR_BGR2HSV)
    mean_H = np.average(temp_sub_img_HSV[:, :, -1])
    if mean_H > 230:
        return True, mean_H
    else:
        return False, mean_H

def get_placeholder():
    images_ph = tf.placeholder(tf.float32, shape=(None, H, W, C))
    labels_ph = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))
    return images_ph, labels_ph


def dropout_train(name):
    print('加载数据....')
    whole_data_path = 'whole_data_front_' + name + '.npy'
    whole_label_path = 'whole_label_' + name + '.npy'
    print('分割数据集')
    train_data, test_data, train_label, test_label = my_train_test_split(whole_data_path, whole_label_path,5)

    summary = [0, 0, 0, 0, 0]
    for label in train_label:
        summary[np.argmax(label)] = summary[np.argmax(label)] + 1
    print(summary)
    images_ph, labels_ph = get_placeholder()
    keep_prob = tf.placeholder(tf.float32)  #drop out 的丢弃值
    logits = NN.get_logits(images_ph, OUTPUT_SIZE,keep_prob)
    #loss = NN.huber_loss(logits, labels_ph)
    # 交叉熵损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_ph,logits=logits))
    #准确率
    correct_pre = tf.equal(tf.argmax(logits,axis=1),tf.argmax(labels_ph,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
    train_op = NN.get_train_op(loss, LEARNING_RATE)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    with tf.Session(config=session_config) as sess:
        sess.run(init_op)

        if os.path.exists(path='ckpt/model.xmly1.ckpt.index'):
            print('restore data from all.ckpt....')
            saver.restore(sess, 'ckpt/model.xmly1.ckpt')
        else:
            print('the first time train')


        def train_batch(batch_data, batch_label):
            feed_dict = {
                images_ph: (np.array(batch_data) / 255.0).astype(np.float64),
                labels_ph: np.array(batch_label).astype(np.int32),
                keep_prob:0.5,
                }
                # print(batch_label)
            _, logits_value,loss_value = sess.run([train_op, logits,loss], feed_dict=feed_dict)
            #print("logits:")
            #print(logits_value)
            #print(logits_value.shape)
            return loss_value

        def train_epoch():
            batch_data = []
            batch_label = []
            for i, index in enumerate(np.random.permutation(len(train_data))):
                batch_data.append(train_data[index])
                batch_label.append(train_label[index])

                if len(batch_data) == BATCH_SIZE:
                        # Train the model
                    loss_value = train_batch(batch_data, batch_label)
                        # Clear old batch
                    batch_data = []
                    batch_label = []

                if i != 0 and i % 500 == 0:
                    print("Step %d: loss %03f" % (i, loss_value))

            if len(batch_data) != 0:
                train_batch(batch_data, batch_label)

        mean_loss_list = []
        mean_acc_list = []
        #mean_train_loss_list = []
        #step = 0
        for i in range(EPOCH_NUM ):
            print("Epoch No.%d"%i)
            train_epoch()
            save_path = saver.save(sess, 'ckpt/model.' + name +'_epoch'+str(i)+ '.ckpt')
            print("Model saved in path: %s" % save_path)
            print('-------在测试集上进行验证----------')
            pre_loss_list = []
            acc_list = []
            for id,one_test_data in enumerate(test_data):
                one_test_data = one_test_data[np.newaxis,:]
                #print(one_test_data.shape)
                feed_dict = {
                    images_ph: (one_test_data/255.0).astype(np.float64),
                    labels_ph: np.array(test_label[id][np.newaxis,:]).astype(np.int32),
                    keep_prob:1.0,
                }
                test_loss,acc = sess.run([loss,accuracy] ,feed_dict=feed_dict)

                pre_loss_list.append(test_loss)
                acc_list.append(acc)

            mean_loss = np.mean(np.array(pre_loss_list))
            mean_acc = np.mean(np.array(acc_list))
            print('mean predict  loss: %f,mean acc:%f' % (mean_loss,mean_acc))
            mean_loss_list.append(mean_loss)
            mean_acc_list.append(mean_acc)
            step = i+1
            if mean_loss_list[i] > mean_loss_list[i-1] and mean_loss_list[i-1] > mean_loss_list[i-2]:#连续2次损失上升停止训练
                print('训练%d次后，出现过拟合，停止训练'%(i-2))
                shutil.copyfile('ckpt/model.' + name +'_epoch'+str(i-2)+ '.ckpt'+'.data-00000-of-00001','ckpt/model.' + name +'.ckpt'+'.data-00000-of-00001')
                shutil.copyfile('ckpt/model.' + name + '_epoch' + str(i - 2) + '.ckpt' + '.index','ckpt/model.' + name + '.ckpt' + '.index')
                shutil.copyfile('ckpt/model.' + name + '_epoch' + str(i - 2) + '.ckpt' + '.meta','ckpt/model.' + name + '.ckpt' + '.meta')
                break

        saver.save(sess,'ckpt/model.'+name+'.ckpt')
        print("各个批次训练后在测试集的平均准确率为 ：")
        print(mean_acc_list)

        plt.xlabel('epoch')
        plt.ylabel('loss and acc')
        label = ['mean loss','mean acc']
        plt.plot(range(step),mean_loss_list,'r')
        plt.plot(range(step), mean_acc_list, 'g')
        plt.legend(label,loc =0 ,ncol =2)
        plt.show()



#评估训练好的某个模型A在测试集B上的准确率,path为训练好的模型ckpt，比如‘ckpt/model.all_epoch9.ckpt’
def stat_acc(model_name='all',data_set = 'testdata'):
    print('加载测试数据')
    test_data = np.load('whole_data_front_'+data_set+'.npy')
    test_label = np.load('whole_label_'+data_set+'.npy')
    print('加载完毕')
    images_ph, labels_ph = get_placeholder()
    keep_prob = tf.placeholder(tf.float32)
    logits = NN.get_logits(images_ph, OUTPUT_SIZE, keep_prob)
    # 准确率
    correct_pre = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels_ph, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        path = 'ckpt/model.'+model_name +'.ckpt'
        saver.restore(sess, path)
        acc_list = []
        for id, one_test_data in enumerate(test_data):
            one_test_data = one_test_data[np.newaxis, :]
            # print(one_test_data.shape)
            feed_dict = {
                images_ph: (one_test_data / 255.0).astype(np.float64),
                labels_ph: np.array(test_label[id][np.newaxis, :]).astype(np.int32),
                keep_prob: 1.0,
            }
            _,acc = sess.run([logits,accuracy], feed_dict=feed_dict)
            acc_list.append(acc)

        mean_acc= np.mean(np.array(acc_list))
        print('训练模型在测试集的平均准确率为：%f'%mean_acc)
    return mean_acc


#已经废弃,不在使用
def train(name):
    train_data = np.load('whole_data_front_' + name + '.npy')
    train_label = np.load('whole_label_' + name + '.npy')
    '''
    if name == 'all':
        files = os.listdir()
        train_data_name_list = list(filter(lambda x: 'whole_data_front_' in x, files))
        name_list = [x[len('whole_data_front_'):-len('.npy')] for x in train_data_name_list]
        print('Train data name list', train_data_name_list)
        train_data = []
        train_label = []
        for name in name_list:
            print(name)
            train_data.append(np.load('whole_data_front_'+name+'.npy'))
            train_label.append(np.load('whole_label_'+name+'.npy'))
        train_data = np.concatenate(train_data, axis=0)
        train_label = np.concatenate(train_label, axis=0)
        name = 'all'
    else:
        train_data = np.load('whole_data_front_'+name+'.npy')
        train_label = np.load('whole_label_'+name+'.npy')
    '''

    train_data = np.load('whole_data_front_' + name + '.npy')
    train_label = np.load('whole_label_' + name + '.npy')

    summary = [0,0,0,0,0]
    for label in train_label:
        summary[np.argmax(label)] = summary[np.argmax(label)]+1
    print(summary)

    images_ph, labels_ph = get_placeholder()
    keep_prob = tf.placeholder(tf.float32)
    logits = NN.get_logits(images_ph, OUTPUT_SIZE,keep_prob)
    #loss = NN.huber_loss(logits, labels_ph)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=images_ph,logits=logits))
    train_op = NN.get_train_op(loss, LEARNING_RATE)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    with tf.Session(config=session_config) as sess:
        sess.run(init_op)

        if os.path.exists(path='ckpt/model.all.ckpt.index'):
            print('restore data from all.ckpt....')
            saver.restore(sess, 'ckpt/model.all.ckpt')
        else:
            print('the first time train')

        def train_batch(batch_data, batch_label):
            feed_dict = {
                images_ph: (np.array(batch_data)/255.0).astype(np.float64),
                labels_ph: np.array(batch_label).astype(np.int32),
                keep_prob:0.5,
            }
            #print(batch_label)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            return loss_value


        '''
        try:
            saver.restore(sess, "ckpt/model."+name+".ckpt")
        except:
            saver.restore(sess,"ckpt/model.ckpt")
        '''

        def train_epoch():
            batch_data = []
            batch_label = []
            for i, index in enumerate(np.random.permutation(len(train_data))):
                batch_data.append(train_data[index])
                batch_label.append(train_label[index])

                if len(batch_data) == BATCH_SIZE:
                    # Train the model
                    loss_value = train_batch(batch_data, batch_label)
                    # Clear old batch
                    batch_data = []
                    batch_label = []

                if i!=0 and i%500 == 0:
                    print("Step %d: loss %03f"%(i, loss_value))

            if len(batch_data) !=0:
                train_batch(batch_data, batch_label)

        for i in range(EPOCH_NUM):
            print("Epoch No.%d"%i)
            train_epoch()
            save_path = saver.save(sess, 'ckpt/model.'+name+'.ckpt')
            print("Model saved in path: %s" % save_path)

        Auto_train.write_train_label_to_file(name,os.getcwd())

#已经废弃不在使用
def auto_train():
    track_name_list  = Auto_train.get_to_train_track_name(os.getcwd())
    if len(track_name_list) ==0:
        print("不需要训练")
    else:
        for track_name in track_name_list:
            print("开始训练%s..."%track_name)
            dropout_train(track_name)
            print('%s 训练完成...'%track_name)
            #重置模型
            tf.reset_default_graph()

#自动训练whole_label_all.npy 文件，根据md5进行校验是否需要训练
def auto_track_all():
    with open('all_md5.txt','r') as f:
        file_md5 = f.read()
    if os.path.exists("whole_label_all.npy"):
        lable_md5 = Auto_train.calc_md5("whole_label_all.npy")
        print(lable_md5)
        if lable_md5 != file_md5:
            print('whole_label_all 需要重新训练')
            dropout_train('all')
            with open('all_md5.txt','w') as f_w:
                f_w.write(lable_md5)

        else:
            print("whole_label_all模型没有变化，不需要训练")
    else:
        print('没有all模型可以训练')

def get_gray_img(img):
    '''1~5, 16~23
    sub_img = img[70:180, 830: 940]
    '''
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

def predict(name):
    Device_Input = Get_Image(port=1313)
    controller = Controller(port=1111) 
    last_4_images = []

    print('-----------------')    
    tf.reset_default_graph()
    images_ph, labels_ph = get_placeholder()
    keep_prob = tf.placeholder(tf.float32)
    logits = NN.get_logits(images_ph, OUTPUT_SIZE, keep_prob)
    saver = tf.train.Saver()
    print('-----------------')  
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, 'ckpt/model.'+name+'.ckpt')
        print("Model restored.")
        

        while True:
            last_time = time.time()

            img = Device_Input.Get_Frame_for_Agent()
            sub_img = get_gray_img(img)
            # 小喷
            whiff,mHean =get_whiff_idx(img)
            last_4_images.append(sub_img)
            
            if len(last_4_images) > C:
                last_4_images.pop(0)
            if len(last_4_images) != C:
                continue

            input_one_record = np.dstack(
            (
                last_4_images[-4].ravel(),
                last_4_images[-3].ravel(),
                last_4_images[-2].ravel(),
                last_4_images[-1].ravel()
            )
            ).ravel().reshape(1, H, W, C)

            feed_dict = {
                images_ph: (np.array(input_one_record)/255.0).astype(np.float64),
                keep_prob:1.0,
            }
            policy = sess.run(logits, feed_dict=feed_dict)
            #[[0.92539525, 0.  , 0. , 0.  , 0. ]]
            #TODO:

            policy = policy * (policy > 0.2)
            action = np.argmax(policy)
            controller.take_action(action_type=action)
            print('Take Action: %d'%action, policy)
            #小喷
            if whiff :
                controller.take_action(action_type=5)
                print('Take action :whiff,%.3f'%mHean)
            cv2.imshow('full image', img)
            cv2.imshow('map', (input_one_record[0,:,:,1:]*255).astype(np.uint8))
            cv2.waitKey(1)

            reverse_flag = test_reverse(img)
            if reverse_flag:
                print("__________________RESET_________________")
            current_time = time.time()
            sleep_time = TIME_GAP - (current_time-last_time)
            if sleep_time > .0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description='PPGAME supervised process train data')
    parser.add_argument('--track', default=' ', help='Track Name')
    parser.add_argument('--type', default='train', help='Train or Test')
    args = parser.parse_args()
    # TODO
    name=args.track
    if args.type == 'Train':
        dropout_train(name)
    elif args.type == 'Auto_train':
       auto_track_all()
    elif args.type == 'Test':
        predict(name)
    else:
        print("Error: type must be Train 、Auto_train、or Test")


if __name__=='__main__':
    main()
    #dropout_train('sd20')
    #stat_acc()
