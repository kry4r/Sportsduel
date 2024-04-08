import taichi as ti
import numpy as np
import cv2
import taichi.math as tm
import csv
import pandas as pd
import math as m
import random

ti.init(arch=ti.gpu)


# @ti.func
def get_angle(v1: np.array, v2: np.array) -> float:
    return (m.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180) / m.pi


# @ti.data_oriented
class poseaccessorpush(object):

    def __init__(self, class_name, left_arm, right_arm, left_forearm, right_forearm) -> None:
        self.class_name = class_name
        self.list = []
        self.all_list = []
        self.acc = []
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.left_forearm = left_forearm
        self.right_forearm = right_forearm
        self.res = -1

    # @ti.func
    def caculate(self) -> ti.float32:
        # Load the normalized pose landmarks.

        # 从pose_landmarks中获取关键点
        if (len(self.left_arm) > len(self.list)):
            diff = len(self.left_arm) - len(self.list)
            for i in range(diff):
                r = int(np.random.uniform(low=0, high=len(self.list) - 1, size=1))
                # r = int(ti.random()*(len(self.list)-1))
                self.list.insert(r, self.list[r])
        else:
            diff = len(self.list) - len(self.left_arm)
            for i in range(diff):
                r = int(np.random.uniform(low=0, high=len(self.right_arm) - 1, size=1))
                # TODO::把这些个变成np.array
                self.left_arm.insert(r, self.left_arm[r])
                self.right_arm.insert(r, self.right_arm[r])
                self.left_forearm.insert(r, self.left_forearm[r])
                self.right_forearm.insert(r, self.right_forearm[r])

        # 计算关键点标准度
        """for i in range(len(pose_list)):
            v1 = tm.vec3(self.landmarks[pose_list[i][0]].x-self.landmarks[pose_list[i][1]].x
                         ,self.landmarks[pose_list[i][0]].y-self.landmarks[pose_list[i][1]].y
                         ,self.landmarks[pose_list[i][0]].z-self.landmarks[pose_list[i][1]].z)
            v2 = tm.vec3(temp_list[pose_list[i][0]].x-temp_list[pose_list[i][1]].x,
                        temp_list[pose_list[i][0]].y-temp_list[pose_list[i][1]].y,
                        temp_list[pose_list[i][0]].z-temp_list[pose_list[i][1]].z)
            """
        # 计算两个向量的夹角
        for i in range(len(self.list)):
            v1 = np.array([self.list[i][14][0] - self.list[i][16][0],
                           self.list[i][14][1] - self.list[i][16][1],
                           self.list[i][14][2] - self.list[i][16][2]])

            v2 = np.array([self.right_forearm[i][0], self.right_forearm[i][1], self.right_forearm[i][2]])
            acc = get_angle(v1, v2)
            self.all_list.append(acc)

        return np.mean(self.all_list)

    # 计数存储
    # @ti.kernel
    def countif(self, now_count: ti.field, count: int, landmark: float) -> ti.float32:
        if (count - now_count[0] == 1):
            # self.list = np.array(self.list)
            self.res = self.caculate()
            self.list = []
            self.all_list = []
            self.acc = []
            now_count[0] += 1
        else:
            # TODO:解决numpy的insert问题
            self.list.append(landmark)

        return self.res