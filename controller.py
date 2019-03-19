"""
Usage:
# To test whether the andriod device is ready
adb shell /data/local/tmp/minitouch -h

# Run minitouch on andrioid devices
adb shell /data/local/tmp/minitouch

# local forward minitouch socket to port #1111
adb forward tcp:1111 localabstract:minitouch
"""
import os
import sys
import math
import time
import socket
import numpy as np


class Controller():
    def __init__(self, port=1111):

        self.port = port
        os.popen('adb shell /data/local/tmp/minitouch')

        time.sleep(1.0)

        os.system('adb forward tcp:%d localabstract:minitouch'%self.port)
        time.sleep(1.0)
        BUF_SIZE = 1024
        server_addr = ('127.0.0.1', self.port)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(server_addr)
        data = self.client.recv(BUF_SIZE)

        self.zero_flag = False
        self.one_flag = False



    def take_action(self, **kargs):

        if kargs['action_type'] ==  0:
            command = 'c\n'
            if self.zero_flag:
                self.zero_flag = False
                command = 'u 0\n'+command
            if self.one_flag:
                self.one_flag = False
                command = 'u 1\n'+command
            if len(command)>2:
                self.client.sendall(bytearray(command, encoding='ascii'))
            
        elif kargs['action_type'] ==  1: #左转
            command = 'd 0 %d %d 50\nc\n'%(200, 300)
            if self.zero_flag:
                self.zero_flag = False
                command = 'u 0\n'+command
            if self.one_flag:
                self.one_flag = False
                command = 'u 1\n'+command
            self.zero_flag = True
            print('turn left')
            self.client.sendall(bytearray(command, encoding='ascii'))
        elif kargs['action_type'] ==  2: #左飘
            command = 'u 0\nd 1 %d %d 50\n'%(200, 1750)+\
                      'd 0 %d %d 50\n'%(200, 300)+\
                      'c\n'
            if self.zero_flag:
                self.zero_flag = False
                command = 'u 0\n'+command
            if self.one_flag:
                self.one_flag = False
                command = 'u 1\n'+command
            self.zero_flag = True
            self.one_flag = True
            print('left shift')
            self.client.sendall(bytearray(command, encoding='ascii'))
        elif kargs['action_type'] ==  3: #右转
            command = 'd 0 %d %d 50\nc\n'%(200, 600)
            if self.zero_flag:
                self.zero_flag = False
                command = 'u 0\n'+command
            if self.one_flag:
                self.one_flag = False
                command = 'u 1\n'+command
            self.zero_flag = True
            print('turn right')
            self.client.sendall(bytearray(command, encoding='ascii'))
        elif kargs['action_type'] ==  4: #右票
            command = 'd 1 %d %d 50\n'%(200, 1750)+\
                      'd 0 %d %d 50\n'%(200, 600)+\
                      'c\n'
            if self.zero_flag:
                self.zero_flag = False
                command = 'u 0\n'+command
            if self.one_flag:
                self.one_flag = False
                command = 'u 1\n'+command
            self.zero_flag = True
            self.one_flag = True
            print('right shift')
            self.client.sendall(bytearray(command, encoding='ascii'))
        elif kargs['action_type'] == 5:  # 小喷
            command = 'd 2 %d %d 50\nc\nu 2\nc\n '% (355, 1440)
            print('whiff ...')
            self.client.sendall(bytearray(command, encoding='ascii'))


def main():
    controller = Controller(port=1111)
    time.sleep(1)
    controller.take_action(action_type=1)
    time.sleep(1)
    controller.take_action(action_type=2)
    time.sleep(1)
    controller.take_action(action_type=3)
    time.sleep(1)
    controller.take_action(action_type=4)
    time.sleep(1)
    controller.take_action(action_type=5)
    time.sleep(1)
    controller.take_action(action_type=0)

if __name__=='__main__':
    main()

