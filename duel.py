# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'duel.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1468, 854)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.enemy = QtWidgets.QLabel(self.centralwidget)
        self.enemy.setGeometry(QtCore.QRect(80, 100, 431, 641))
        self.enemy.setObjectName("enemy")
        self.self = QtWidgets.QLabel(self.centralwidget)
        self.self.setGeometry(QtCore.QRect(960, 80, 431, 651))
        self.self.setObjectName("self")
        self.start_button = PrimaryPushButton(self.centralwidget)
        self.start_button.setGeometry(QtCore.QRect(630, 190, 153, 32))
        self.start_button.setObjectName("start_button")
        self.Time = TimeEdit(self.centralwidget)
        self.Time.setGeometry(QtCore.QRect(630, 140, 151, 33))
        self.Time.setObjectName("Time")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1468, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.enemy.setText(_translate("MainWindow", "enemy"))
        self.self.setText(_translate("MainWindow", "self"))
        self.start_button.setText(_translate("MainWindow", "Start"))
from qfluentwidgets import PrimaryPushButton, TimeEdit