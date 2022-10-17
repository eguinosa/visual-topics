# Gelin Eguinosa Rosique
# 2022

import sys
from PyQt6.QtWidgets import (
    QApplication, QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout,
    QWidget
)
from PyQt6.QtCore import Qt


class QUpdatesDialog(QDialog):
    """
    Dialog use to make big changes on the Topic Model of the IR system, like
    changing the Topic Size or the Topic Model used.
    """

    def __init__(
            self, action_text: str, message_text: str,
            parent_widget: QWidget = None
    ):
        """
        Initialize class and attributes.
        """
        # Initialize the base class.
        super().__init__(parent_widget)
        self.setModal(True)

        # Save Class Attributes.
        self.action_text = action_text
        self.message_text = message_text

        # Create the UI.
        self.initializeUI()

    def initializeUI(self):
        """
        Set up the Application's GUI.
        """
        self.setMinimumSize(400, 150)
        self.setWindowTitle(self.action_text)
        self.setUpMainWindow()

    def setUpMainWindow(self):
        """
        Create and arrange widgets in the main window.
        """
        # - Header Label -
        header_label = QLabel(self.message_text)
        # - Progress Bar -
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)
        # - Cancel Button -
        cancel_button = QPushButton("Cancel")
        cancel_button.setDefault(False)
        cancel_button.setAutoDefault(False)
        cancel_button.setEnabled(False)
        cancel_button.clicked.connect(
            lambda checked: self.reject()
        )
        # - Create Main Layout -
        main_v_box = QVBoxLayout()
        main_v_box.addWidget(header_label, 0, Qt.AlignmentFlag.AlignLeft)
        main_v_box.addWidget(progress_bar)
        main_v_box.addWidget(cancel_button, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(main_v_box)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QUpdatesDialog('Food', 'Finding food')
    window.show()
    sys.exit(app.exec())
