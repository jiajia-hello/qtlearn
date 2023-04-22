import sys

from PyQt5 import QtWidgets, QtCore, QtGui

from PyQt5.QtGui import *

from PyQt5.QtWidgets import *

from PyQt5.QtCore import *
from flowerClassfication import apply

class picture(QWidget):

    def __init__(self):
        super(picture, self).__init__()
        self.resize(1200, 800)
        self.setWindowTitle("识别花卉种类")
        self.label = QLabel(self)
        self.label.setText(" 显示图片")
        self.label.setFixedSize(500, 400)
        self.label.move(160, 150)

        self.area = QLabel(self)
        self.area.setText("识别结果：")
        self.area.setFixedSize(300, 100)
        self.area.move(700, 150)

        self.p_area = QLabel(self)
        self.p_area.setFixedSize(150, 100)
        self.p_area.move(1000, 150)

        self.label.setStyleSheet("QLabel{background:white;}"
        
        "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"

        )
        self.area.setStyleSheet("QLabel{background:white;}"

                                 "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"

                                 )
        self.p_area.setStyleSheet("QLabel{background:white;}"

                                "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"

                                )

        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(200, 600)
        btn.clicked.connect(self.openimage)

        clear_btn = QPushButton(self)
        clear_btn.setText("清除")
        clear_btn.move(350,600)
        clear_btn.clicked.connect(self.clear)

    # def openimage(self):
    #     imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
    #     jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
    #     self.label.setPixmap(jpg)

    def clear(self):
        self.area.clear()
        self.label.clear()
        self.p_area.clear()
        self.label.setText(" 显示图片")
        self.area.setText("识别结果：")


    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        result, plist = apply(imgName)
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        print('pred:',result)
        name = ""
        probablity = ""
        for item, p in zip(result,plist):
            name += item+"\n"
            probablity += str(round(p,4))+"\n"
        self.area.setText(name)
        self.p_area.setText(probablity)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
