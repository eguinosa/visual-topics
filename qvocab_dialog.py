# Gelin Eguinosa Rosique
# 2022

import sys
from PyQt6.QtWidgets import (
    QApplication, QDialog, QLabel, QTextEdit, QPushButton, QVBoxLayout,
)
from PyQt6.QtCore import Qt


class QVocabDialog(QDialog):

    def __init__(self, topic_id: str, word_list: list):
        """
        Initialize class and Attributes.
        """
        # Initialize the base class.
        super().__init__()
        self.setModal(False)

        # Save the class attributes.
        self.topic_id = topic_id
        self.word_list = word_list

        # Create the UI
        self.initializeUI()

    def initializeUI(self):
        """
        Set up the Application's GUI.
        """
        self.setMinimumSize(640, 420)
        self.setWindowTitle(f"Vocabulary - {self.topic_id}")
        self.setUpMainWindow()

    def setUpMainWindow(self):
        """
        Create and arrange widgets in the main window.
        """
        # Text for the Header.
        word_count = len(self.word_list)
        header_text = f"""
            <p align='left'><font face='Times New Roman' size='+1'><u>
            {word_count} Top Words from {self.topic_id}:
            </u></font></p>
        """
        # Text for Vocabulary.
        vocab_text = ''
        is_first = True
        for word, sim in self.word_list:
            # # Check if this is the first element.
            if not is_first:
                vocab_text += '\n\n'
            vocab_text += f"""<p>({word}, {round(sim, 5)})</p>"""
            # Flag the rest of the elements.
            if is_first:
                is_first = False

        # Create Labels, Widget & Areas.
        header_label = QLabel(header_text)
        vocab_read_only = QTextEdit(vocab_text)
        vocab_read_only.setReadOnly(True)
        done_button = QPushButton('Done')
        # noinspection PyTypeChecker
        done_button.clicked.connect(self.close)

        # Create the Main Layout.
        main_v_box = QVBoxLayout()
        main_v_box.addWidget(header_label)
        main_v_box.addWidget(vocab_read_only)
        main_v_box.addWidget(done_button, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(main_v_box)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QVocabDialog(
        topic_id='Topic_2',
        word_list=[
            ('rose', 0.233), ('flower', 0.333), ('mask', 0.232),
        ]
    )
    window.show()
    sys.exit(app.exec())
