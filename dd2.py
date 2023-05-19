from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtGui import QFont
from PyQt5.uic import loadUi

class PasswordUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Lock Screen')
        self.setGeometry(300, 300, 300, 150)
        self.setFont(QFont('Arial', 12))

        # 암호 입력 위젯
        self.password_label = QLabel('Enter Password:', self)
        self.password_label.move(20, 20)

        self.password_input = QLineEdit(self)
        self.password_input.move(20, 50)

        # 잠금 해제 버튼
        self.unlock_button = QPushButton('Unlock', self)
        self.unlock_button.move(20, 80)
        self.unlock_button.clicked.connect(self.check_password)

        # 수직 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.unlock_button)

        self.setLayout(layout)

        # 다른 UI 위젯
        self.ui_widget = None

    # 암호 확인 함수
    def check_password(self):
        password = self.password_input.text()
        if password == '1234':  # 올바른 암호
            if not self.ui_widget:
                self.ui_widget = loadUi('./testBtn.ui', self)  # self.ui_widget에 로드한 위젯 할당
            self.ui_widget.show()
            self.hide()  # PasswordUI 위젯 숨기기
        else:  # 잘못된 암호
            self.password_input.clear()
            self.password_input.setPlaceholderText('Incorrect password')


if __name__ == '__main__':
    app = QApplication([])
    password_ui = PasswordUI()
    password_ui.show()
    app.exec()