import os
import sys

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
import qimage2ndarray
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

import deepfake_detection

form_class = uic.loadUiType("deepfakeGUI_ch3.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.label = self.findChild(QLabel, "photo")

        self.answerbtn = self.findChild(QPushButton, "answerbtn")
        self.cropimg = self.findChild(QLabel, "cropimg")
        self.gradimg = self.findChild(QLabel, "gradimg")



    def realclick(self):
        # return basic
        self.answerbtn.setText("결과")

        self.gradimg.clear()
        self.cropimg.clear()
        self.label.clear()
        real_fname = QFileDialog.getOpenFileName(self, "Open File", "C:\\RealImage",
                                                 "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")

        # Open The Image
        self.pixmap = QPixmap(real_fname[0])
        # Add Pic to label
        self.label.setPixmap(self.pixmap)
        result, cropimg, gradcam = deepfake_detection.deepfake_detection(real_fname[0])
        self.resultbtn.clicked.connect(lambda: self.resultclick(result, cropimg, gradcam))

    def fakeclick(self):
        # return basic
        self.answerbtn.setText("결과")

        self.gradimg.clear()
        self.cropimg.clear()
        self.label.clear()
        fake_fname = QFileDialog.getOpenFileName(self, "Open File", "C:\\FakeImage",
                                                 "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")

        # Open The Image
        self.pixmap = QPixmap(fake_fname[0])
        # Add Pic to label
        self.label.setPixmap(self.pixmap)
        result, cropimg, gradcam = deepfake_detection.deepfake_detection(fake_fname[0])
        self.resultbtn.clicked.connect(lambda: self.resultclick(result, cropimg, gradcam))

    def resultclick(self, result, crop, grad):
        # text result
        self.answerbtn.setText(result)
        # show crop image
        crop_var = qimage2ndarray.array2qimage(crop, normalize=False)
        self.croppixmap = QPixmap(crop_var)
        self.cropimg.setPixmap(self.croppixmap)
        # show grad image
        grad_var = qimage2ndarray.array2qimage(grad, normalize=False)
        self.gradpixmap = QPixmap(grad_var)
        self.gradimg.setPixmap(self.gradpixmap)

        # self.result_label.setText(n)

    def finishclick(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()