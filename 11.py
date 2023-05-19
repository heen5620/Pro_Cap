import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class HandUnlockWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Unlock")
        self.setGeometry(100, 100, 300, 300)

        label = QLabel(self)
        pixmap = QPixmap('hand.png')
        label.setPixmap(pixmap)
        label.resize(pixmap.width(), pixmap.height())
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HandUnlockWindow()
    window.show()
    sys.exit(app.exec_())
