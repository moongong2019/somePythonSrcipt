import os
import re
import NN
import sys
import cv2
import time
import signal
import argparse
import numpy as np
import multiprocessing
import tensorflow as tf
from get_image import Get_Image
from controller import Controller
from preprocessor import Preprocessor
from test_reverse import test_reverse, test_small_injection, test_big_injection


H = 84
W = 84
C = 4
OUTPUT_SIZE = 5
PORT = 23333
TIME_GAP = 0.1
SAVE_TIME_GAP = 1.0
SAVE_INTERVAL = int(SAVE_TIME_GAP/TIME_GAP)


def get_device_no():
    r = os.popen('adb devices')
    info = r.readlines()
    device_list = []
    for line in info:
        line = line.strip('\r\n')
        if len(line)>2 and '\tdevice' in line:
            device_list.append(line.split('\t')[0])
    print('Device list: %s'%str(device_list))
    return device_list

def get_screen_shape(device_list):
    display_shape_list = []
    for device_no in device_list:
        r = os.popen('adb -s %s shell dumpsys window displays | find /i "init"'%(device_no,))
        info = r.readlines()[0]
        for s in info.split(' '):
            if 'init=' in s:
                display_shape_list.append(re.findall(r"\d+\.?\d*",s))
    print('Display shape:', display_shape_list)
    return display_shape_list

def get_config_ready():
    device_list = get_device_no()
    display_shape_list = get_screen_shape(device_list)
    # Get port list:
    port_list = [i for i in range(PORT, PORT+2*len(device_list))]
    print('Following port will be occupied %s'%str(port_list))
    # Get config list:
    config_list = []
    for idx, (device_no, display_shape) in enumerate(zip(device_list, display_shape_list)):
        config_list.append(
            {
              'device_no': device_no,
              'display_shape': display_shape,
              'device_input': Get_Image(device_no, port_list[2*idx], display_shape),
              'controller':Controller(device_no, port_list[2*idx+1], display_shape),
              'preprocessor':Preprocessor(display_shape),
            })

    # Wait until all config finished
    time.sleep(2)
    return config_list, device_list

def process_kenerl(undecode_image):
    image = cv2.imdecode(np.array(bytearray(undecode_image)), 1)
    h, w, _ = image.shape
    image = cv2.resize(image, (2*w, 2*h))
    return image

def get_one_step_data(config_list, pool):
    img_list = []
    sub_img_list = []
    speed_list = []
    reset_flag_list = []

    undecode_image_list = []
    for config in config_list:
        undecode_image_list.append(config['device_input'].Get_Frame_for_Agent_undecode())
    results = []
    for undecode_image in undecode_image_list:
        result = pool.apply_async(process_kenerl, args=(undecode_image,))
        results.append(result)

    #pool.close()
    #pool.join()

    for result in results:
        img_list.append(result.get())

    for config, image in zip(config_list, img_list):
        sub_img, speed, reset_flag = config['preprocessor'].process(image)
        sub_img_list.append(sub_img)
        speed_list.append(speed)
        reset_flag_list.append(reset_flag)

    #将多台机器的小地图信息存入到sub_img
    sub_img = np.array(sub_img_list).reshape(-1, H, W, C)

    return sub_img, speed_list, reset_flag_list, img_list

def save_image_kenerl(img, device_no, save_image_path):
    image_path = os.path.join(save_image_path, '%s_%s.jpg'%(device_no, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    cv2.imwrite(image_path, img)

def injection_reverse_kenerl(img, controller, reset_flag, step):
    small_injection_flag, _ = test_small_injection(img)
    if small_injection_flag:
        controller.take_action(action_type='click', x=355, y=1440)

    # 13 is chosen randomly.
    if step%13==0:
        big_injection_flag, _ = test_big_injection(img)
        if big_injection_flag:
            controller.take_action(action_type='click', x=460, y=1630)

    if reset_flag:
        controller.take_action(action_type='reset')
    # 7 is chosen randomly.
    elif step%7==0:
        reverse_flag = test_reverse(img)
        if reverse_flag:
            controller.take_action(action_type='reset')


def run_one_game(sess, config_list, device_list, images_ph, logits, save_image_path, pool,keep_prob):
    step = 0
    while True:
        step+=1

        last_time = time.time()
        sub_img, speed_list, reset_flag_list, img_list = get_one_step_data(config_list, pool)
        feed_dict = {
                images_ph: (sub_img/255.0).astype(np.float32),
                keep_prob:1.0,
            }
       #对多台机器的小地图输入神经网络进行预测
        policy = sess.run(logits, feed_dict=feed_dict)
        policy = policy * (policy > 0.2)
        #返回每台机器的小地图的预测操作
        action_list = np.argmax(policy, axis=1)
        #print("take action %d"%action_list)

        # Take action part:
        for action, config in zip(action_list, config_list):
            config['controller'].take_action(action_type=action)
        # Show result part:
        """
        info = ''
        for device_no, action, speed, reset in zip(device_list, action_list, speed_list, reset_flag_list):
            info = info+'Device:%s, speed:%d, action:%d, reset:%s\t'%(device_no, speed, action, str(reset))
            speed_last_10.append(speed)
            speed_last_10.pop(0)
        print(info)
        """
        show_img = sub_img.reshape(-1, W, C)
        cv2.imshow('show', show_img)
        cv2.waitKey(1)

        # injection / reverse / reset:
        for img, config, reset_flag in zip(img_list, config_list, reset_flag_list):
            pool.apply_async(injection_reverse_kenerl, args=(img, config['controller'], reset_flag, step))

        # save image:
        if save_image_path and (step%SAVE_INTERVAL==0):
            for img, device_no in zip(img_list, device_list):
                pool.apply_async(save_image_kenerl, args=(img, device_no, save_image_path))

        current_time = time.time()
        sleep_time = TIME_GAP-(current_time-last_time)
        if sleep_time > .0:
            time.sleep(sleep_time)


def get_placeholder():
    images_ph = tf.placeholder(tf.float32, shape=(None, H, W, C))
    labels_ph = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))
    return images_ph, labels_ph

def main(trackname, save_image_path):
    '''
    For multi processes speed up
    '''
    device_list = get_device_no()
    num_processes = len(device_list)
    # Add following line in Windows plateform, to avoid RuntimeError
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(processes=num_processes)
    # Init opencv window
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    config_list, device_list = get_config_ready()

    # For multi thread clear
    def Handler(sig_num, frame):
        print("EXIT!!! CLOSE CONNECTION!!")
        pool.close()
        pool.join()
        for config in config_list:
            config['controller'].close()
            config['device_input'].close()
        sys.exit(sig_num)
    signal.signal(signal.SIGINT, Handler)

    print('---------INIT TENSORFLOW----------')
    tf.reset_default_graph()
    images_ph, labels_ph = get_placeholder()
    keep_prob = tf.placeholder(tf.float32)
    logits = NN.get_logits(images_ph, OUTPUT_SIZE,keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "ckpt/model."+trackname+".ckpt")
        print("Model %s restored."%("ckpt/model."+trackname+".ckpt",))

        print("Press b to begin game")
        # Wait human players enter the game.
        while True:
            sub_img, speed_list, reset_flag_list, img_list = get_one_step_data(config_list, pool)
            show_img = sub_img.reshape(-1, W, C)
            cv2.imshow('show', show_img)
            if cv2.waitKey(1) & 0xFF == ord('b'):
                break
        print("------------GAME START---------------")
        # Enter the game, and control
        run_one_game(sess, config_list, device_list, images_ph, logits, save_image_path, pool,keep_prob)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PPGAME yeti multi device control')
    parser.add_argument('--track', default=' ', help='Track Name')
    parser.add_argument('--save_image_path', default=None, help='The path to save the image of game per second')
    args = parser.parse_args()
    name = args.track
    '''
    if args.track in ['xscz', 'hbhc', 'dbfrq', 'tkpshp', 'xjdhs',
    		 'czwlcc', 'hdsd','czgsgl', 'hdjbhw', 'jxfjc','smjzt', 'kljdc']:
        main(args.track)
    else:
        print(args.track, 'not trained')
    '''
    # Create dir for image
    if args.save_image_path:
        if not os.path.exists(args.save_image_path):
            os.makedirs(args.save_image_path)

    main(name, args.save_image_path)
