# Gelin Eguinosa Rosique
# 2022

import sys
from random import choice
from PyQt6.QtWidgets import (
    QApplication, QDialog, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
    QDialogButtonBox, QWidget
)
from PyQt6.QtCore import Qt


class QOpenDocument(QDialog):
    """
    Class to get the ID of the Document we have to open.
    """

    def __init__(self, all_doc_ids: list, parent_widget: QWidget = None):
        """
        Initialize class and attributes.
        """
        # Initialize the base class.
        super().__init__(parent_widget)
        self.setModal(True)

        # Save Class Attributes.
        self.doc_ids = all_doc_ids
        self.text = ''
        self.feedback_label = None
        self.old_match = ''
        self.doc_id = ''

        # Create the UI.
        self.initializeUI()

    def initializeUI(self):
        """
        Set up the Application's GUI.
        """
        self.setMinimumSize(400, 150)
        self.setWindowTitle("Open Document")
        self.setUpMainWindow()

    def setUpMainWindow(self):
        """
        Create and arrange widgets in the main window.
        """
        # ID Widgets.
        id_label = QLabel("Doc ID:")
        id_edit = QLineEdit()
        id_edit.setPlaceholderText(" ID of the Document")
        id_edit.textEdited.connect(
            lambda new_text: self.textEdited(new_text=new_text)
        )
        edit_layout = QHBoxLayout()
        edit_layout.addWidget(id_label, 0, Qt.AlignmentFlag.AlignLeft)
        edit_layout.addWidget(id_edit)
        # Feedback Label.
        feedback_label = QLabel(f"[INFO] Try <{choice(self.doc_ids)}>")
        self.feedback_label = feedback_label
        feedback_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        # Accept & Cancel.
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        button_box.accepted.connect(
            lambda: self.acceptDocID()
        )
        button_box.rejected.connect(
            lambda: self.reject()
        )
        # Other Way for Accept & Cancel.
        # accept_button = QPushButton("Accept")
        # accept_button.setDefault(True)
        # accept_button.clicked.connect(
        #     lambda checked: self.acceptClicked()
        # )
        # cancel_button = QPushButton("Cancel")
        # cancel_button.setCheckable(True)
        # cancel_button.setAutoDefault(False)
        # cancel_button.clicked.connect(
        #     lambda checked: self.reject()
        # )
        # buttons_layout = QHBoxLayout()
        # buttons_layout.addWidget(cancel_button, 0, Qt.AlignmentFlag.AlignBottom)
        # buttons_layout.addWidget(accept_button, 0, Qt.AlignmentFlag.AlignBottom)

        # Create Main Layout.
        main_v_box = QVBoxLayout()
        main_v_box.addLayout(edit_layout)
        main_v_box.addWidget(feedback_label, 0, Qt.AlignmentFlag.AlignLeft)
        main_v_box.addWidget(button_box)
        self.setLayout(main_v_box)

    def textEdited(self, new_text: str):
        """
        The Text Edit Label has new text.
        """
        # Save the text.
        self.text = new_text
        # Check if there is a match.
        if self.old_match and self.old_match.startswith(new_text):
            self.feedback_label.setText(f"[INFO]: Match found <{self.old_match}>")
        if any((example := doc_id).startswith(new_text) for doc_id in self.doc_ids):
            self.old_match = example
            self.feedback_label.setText(f"[INFO]: Match found <{example}>")
        else:
            self.feedback_label.setText("[INFO]: No Match found")

    def acceptDocID(self):
        """
        Analyse the Text of the TextEdit widget to see if it is a valid Doc ID.
        """
        if self.text in self.doc_ids:
            self.doc_id = self.text
            self.accept()
        else:
            self.reject()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QOpenDocument(all_doc_ids=['food202', 'marca283', 'fiesta89', 'fiefo888'])
    window.show()
    sys.exit(app.exec())
