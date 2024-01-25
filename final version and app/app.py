import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget, QFileDialog, QGridLayout
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

from algorithm import main_stitching

widgets = {
    "logo": [],
    "first_message": [],
    "quit_button": [],
    "select_folder_button": [],
    "message": [],
    "result_image": []
}

grid = QGridLayout()

def clear_widgets():
    for widget in widgets:
        if widgets[widget] != []:
            widgets[widget][-1].hide()
        for i in range(0, len(widgets[widget])):
            widgets[widget].pop()

def start_app():
    clear_widgets() 
    first_screen()

def create_buttons(message, l_margin, r_margin):
    button = QPushButton(message)
    button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
    button.setFixedWidth(485)
    button.setStyleSheet(
        "*{margin-left: " + str(l_margin) +"px;"+
        "margin-right: " + str(r_margin) +"px;"+
        '''
        border: 4px solid '#BC006C';
        color: white;
        font-family: 'shanti';
        font-size: 16px;
        border-radius: 25px;
        padding: 15px 0;
        margin-top: 20px;
        text-align: center;
        }
        *:hover{
            background: '#BC006C';
        }
        '''
    )
    #button.setAlignment(QtCore.Qt.AlignCenter)
    return button

def set_logo():
    image = QPixmap("logo_bottom_2.png")
    logo = QLabel()
    logo.setPixmap(image)
    logo.setAlignment(QtCore.Qt.AlignCenter)
    logo.setStyleSheet("margin-top: 75px; margin-bottom: 30px;")
    widgets["logo"].append(logo)

def set_message(msg):
    message = QLabel(msg)
    message.setAlignment(QtCore.Qt.AlignCenter)
    message.setStyleSheet(
        "font-family: 'Shanti'; font-size: 30px; color: 'white'; margin-top:0px; margin-bottom:0px;"
        )
    widgets["message"].append(message)

def first_screen():
    first_message = QLabel("Welcome to the Application\nPlease select a folder...")
    first_message.setAlignment(QtCore.Qt.AlignCenter)
    first_message.setWordWrap(True)
    first_message.setStyleSheet(
        '''
        font-family: 'shanti';
        font-size: 25px;
        color: 'white';
        padding: 75px;
        '''
    )
    widgets["first_message"].append(first_message)


    button1 = create_buttons("Quit", 85, 5)
    button2 = create_buttons("Select a folder", 5, 85)

    widgets["quit_button"].append(button1)
    widgets["select_folder_button"].append(button2)

    button1.clicked.connect(lambda: handle_button_click_1(1))
    button2.clicked.connect(lambda: handle_button_click_2(2))

    set_logo()

    grid.addWidget(widgets["first_message"][-1], 1, 0, 1, 2)
    grid.addWidget(widgets["quit_button"][-1], 2, 0)
    grid.addWidget(widgets["select_folder_button"][-1], 2, 1)
    grid.addWidget(widgets["logo"][-1], 4, 0, 1,2)

def handle_button_click_1(button_number):
    print(f"Button {button_number} clicked")
    sys.exit()

def handle_button_click_2(button_number):
    print(f"Button {button_number} clicked")
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder", QtCore.QDir.homePath())
    if folder_path:
        print("Seçilen Klasör:", folder_path)
        second_screen(folder_path)
    else:
        #TODO: bekleyip ilk sayfaya geri dönsün
        print("Klasör seçimi iptal edildi.")
    
def second_screen(folder_path):

    clear_widgets()

    set_message(f"Images are stitching...\n\n\nYou selected the folder: {folder_path}")
    set_logo()

    grid.addWidget(widgets["message"][-1], 1, 0, 1, 2)
    grid.addWidget(widgets["logo"][-1], 2, 0, 1, 2)
    QtCore.QTimer.singleShot(100, lambda: second_screen_2(folder_path))
    #second_screen_2(folder_path)

def second_screen_2(folder_path):
    can_continue, time, image_folder_path = main_stitching(folder_path)
    if can_continue:
        third_screen(time, image_folder_path)

def third_screen(time, image_folder_path):
    print(f"third_screen {image_folder_path}")
    clear_widgets()

    result_image_file = QPixmap(image_folder_path)
    scaled_pixmap = result_image_file.scaled(1000, 500, Qt.KeepAspectRatio)
    result_image = QLabel()
    result_image.setPixmap(scaled_pixmap)
    result_image.setAlignment(QtCore.Qt.AlignCenter)
    widgets["result_image"].append(result_image)

    set_message(f"Images are stitched. Time: {time} \n The location where it is saved: {image_folder_path}")
    set_logo()

    button1 = create_buttons("Quit", 85, 5)
    button2 = create_buttons("New...", 5, 85)

    widgets["quit_button"].append(button1)
    widgets["select_folder_button"].append(button2)

    button1.clicked.connect(lambda: handle_button_click_1(1))
    button2.clicked.connect(lambda: handle_button_click_2(2))

    grid.addWidget(widgets["message"][-1], 1, 0, 1, 2)
    grid.addWidget(widgets["result_image"][-1], 2, 0, 1, 2)
    grid.addWidget(widgets["quit_button"][-1], 3, 0)
    grid.addWidget(widgets["select_folder_button"][-1], 3, 1)
    grid.addWidget(widgets["logo"][-1], 4, 0, 1, 2)

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Stitching of Vehicle Underbody Images")
#window.setFixedWidth(1000)

#window.move(2700, 200)
window.setStyleSheet("background: #161219;")

start_app()

window.setLayout(grid)
window.showMaximized()

window.show()
sys.exit(app.exec())