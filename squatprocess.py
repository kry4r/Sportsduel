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
class poseaccessorsquat(object):

    def __init__(self, class_name, body, leg) -> None:
        self.class_name = class_name
        self.list = []
        self.all_list = []
        self.acc = []
        self.leg = leg
        self.body = body
        self.leg_save = leg
        self.body_save = body
        self.res = 0

    # @ti.func
    def caculate(self) -> ti.float32:
        # Load the normalized pose landmarks.

        # 从pose_landmarks中获取关键点
        if (len(self.body) > len(self.list)):
            diff = len(self.body) - len(self.list)
            for i in range(diff):
                r = int(np.random.uniform(low=1, high=len(self.list) - 1, size=1))
                # r = int(ti.random()*(len(self.list)-1))
                self.list.insert(r, self.list[r - 1])
        else:
            diff = len(self.list) - len(self.body)
            for i in range(diff):
                r = int(np.random.uniform(low=1, high=len(self.body) - 1, size=1))
                # TODO::把这些个变成np.array
                self.leg.insert(r, self.leg[r - 1])
                self.body.insert(r, self.body[r - 1])

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
            v1 = np.array([self.list[i][11][0] - self.list[i][23][0],
                           self.list[i][11][1] - self.list[i][23][1],
                           self.list[i][11][2] - self.list[i][23][2]])

            v2 = np.array([self.body[i][0], self.body[i][1], self.body[i][2]])
            acc = get_angle(v1, v2)
            self.all_list.append(acc)

        acc = np.mean(self.all_list)
        if (acc <= 30):
            return 0
        return acc

    # 计数存储
    # @ti.kernel
    def countif(self, now_count: ti.field, count: int, landmark: float) -> ti.float32:
        if (count - now_count[0] == 1):
            # self.list = np.array(self.list)
            self.res = self.caculate()
            self.list = []
            self.all_list = []
            self.acc = []
            self.leg = self.leg_save
            self.body = self.body_save
            now_count[0] += 1
        else:
            # TODO:解决numpy的insert问题
            self.list.append(landmark)

        return self.res