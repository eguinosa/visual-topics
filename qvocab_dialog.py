# Gelin Eguinosa Rosique
# 2022

import json
import sys
from os import mkdir
from os.path import isdir, isfile, join

from PyQt6.QtWidgets import (
    QApplication, QDialog, QLabel, QTextEdit, QPushButton, QVBoxLayout,
    QCheckBox, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent


class QVocabDialog(QDialog):
    """
    Class to show in a Dialog the words that best describe the topic of a Topic
    Model.
    """
    # Class Data Locations.
    data_folder = 'project_data'
    class_folder = 'qvocab_dialog_files'
    words_only_file = 'show_words_only.json'

    def __init__(
            self, topic_id: str, word_list: list, parent_widget: QWidget = None
    ):
        """
        Initialize class and Attributes.
        """
        # Initialize the base class.
        super().__init__(parent_widget)
        self.setModal(False)

        # Check if the Class Folder exists.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        class_folder_path = join(self.data_folder, self.class_folder)
        if not isdir(class_folder_path):
            mkdir(class_folder_path)

        # See if we can load the file with the 'words_only' information.
        words_only_path = join(class_folder_path, self.words_only_file)
        if not isfile(words_only_path):
            words_only = False
        else:
            # Load file and save its value.
            with open(words_only_path, 'r') as f:
                words_only = json.load(f)

        # Save the class attributes.
        self.topic_id = topic_id
        self.word_list = word_list
        self.words_only = words_only

        # Widget used to display the text with the vocabulary of the Topic.
        self.vocab_text_widget = None

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
            <p align='left'><font face='Times New Roman' size='+1'>
            {word_count} Top Words from {self.topic_id}:
            </font></p>
        """
        # Create Labels, Widget & Areas.
        header_label = QLabel(header_text)
        vocab_read_only = QTextEdit()
        vocab_read_only.setReadOnly(True)
        self.vocab_text_widget = vocab_read_only
        self.updateVocabText()
        words_checkbox = QCheckBox("Words Only")
        words_checkbox.setChecked(self.words_only)
        words_checkbox.toggled.connect(
            lambda checked: self.wordsOnlyChange(checked=checked)
        )
        done_button = QPushButton('Done')
        # noinspection PyTypeChecker
        done_button.clicked.connect(self.close)

        # Create the Main Layout.
        main_v_box = QVBoxLayout()
        main_v_box.addWidget(header_label)
        main_v_box.addWidget(vocab_read_only)
        main_v_box.addWidget(words_checkbox)
        main_v_box.addWidget(done_button, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(main_v_box)

    def updateVocabText(self):
        """
        Create or Update the Text Displaying the Vocabulary of the Topic,
        depending on the value of the 'words_only' attribute.
        """
        # Create Text.
        vocab_text = ''
        is_first = True
        for word, sim in self.word_list:
            # Add a separator for all non-first elements.
            if not is_first:
                # Use commas when displaying only words.
                if self.words_only:
                    vocab_text += ', '
                # Separate by lines in other cases.
                else:
                    vocab_text += '\n'
            # Add the Word.
            if self.words_only:
                vocab_text += word
            else:
                vocab_text += f"""<p>({word}, {round(sim, 5)})</p>"""
            # Flag the elements after the first.
            if is_first:
                is_first = False

        # Change the Text of the Widget.
        self.vocab_text_widget.setText(vocab_text)

    def wordsOnlyChange(self, checked: bool):
        """
        Update the way the words in the vocabulary of the Topic are displayed,
        and the attribute that stores this state.
        """
        # Update Words Only of class attribute.
        self.words_only = checked
        # Update Vocabulary Text.
        self.updateVocabText()

    def closeEvent(self, event: QCloseEvent):
        """
        Reimplement the closing event to save the 'words_only' variable so we
        have the same behavior the next time we open the dialog.
        """
        # Saving the value of the Words Only variable.
        class_folder_path = join(self.data_folder, self.class_folder)
        words_only_path = join(class_folder_path, self.words_only_file)
        with open(words_only_path, 'w') as f:
            json.dump(self.words_only, f)

        # Accept the Closing Event.
        event.accept()


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
