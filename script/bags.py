#!/usr/bin/env python
import rospy
import numpy
import csv
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32,String
from sensor_msgs.msg import Image

'''
read tum database bag file
attach groundtruth.txt timestamp with images and xyz rpy
save bag image to img/ fold
save timestamp, img, x ,y ,z, r, p , y to train.txt and test.txt
'''


def readfile(gt_file):
    gt_dict = []
    time = 0
    with open(gt_file) as f:
        line = f.readline()
        cnt =0
        while line:
            line = f.readline()
            str_list = line.split()
            if(len(str_list) == 8):
                #print str_list
                if (abs(time - float(str_list[0])) > 0.3):
                    gt_dict.append(str_list)
                    time = float(str_list[0])
    f.close()
    print("finish read txt")
    return gt_dict

def readcsvfile(gt_file):
    with open(gt_file) as csvfile:
        spamreader = csv.reader(csvfile)
        gt_dict = []
        for row in spamreader:
            gt_dict.append(row)
    return gt_dict

if __name__ == '__main__':

    rospy.init_node("task_server")
    gt_file = 'groundtruth.txt'
    image_path = "img/"
    bag = rosbag.Bag('rgbd_dataset_freiburg2_pioneer_slam2.bag')
    dataset = "dataset.txt"
    image_topic = '/camera/rgb/image_color'
    gt_dict = readfile(gt_file)
    #gt_dict = readcsvfile(gt_file)
    min_time = 0.005
    k = 0
    f = open('dataset.txt', 'w')
    trainfile = open('dataset_train.txt', 'w')
    testfile = open('dataset_test.txt', 'w')
    
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        
        if len(gt_dict)>0 and abs((float(t.to_sec()) - float(gt_dict[0][0])) < min_time):
            #save_image
            #print "got time"
            cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
            k += 1
            image_name = image_path + str(k) + ".png"
            #cv2.imwrite(image_name, cv_image)
            cv2.imwrite(image_name, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
            gt_dict[0][0] = image_name
            line_str = ""
            for i in gt_dict[0]:
                line_str += str(i)+ " "
            line_str += "\n"
            f.write(line_str)
            if(k % 10 == 0):
                testfile.write(line_str)
            else:
                trainfile.write(line_str)
            gt_dict.pop(0)
        elif len(gt_dict) == 0:
            print ("finish read all groundturth data")
            break
        #rospy.sleep(0.1)
        
    f.close()
    testfile.close()
    trainfile.close()
    bag.close()
