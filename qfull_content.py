# Gelin Eguinosa Rosique
# 2022

import sys
from PyQt6.QtWidgets import (
    QApplication, QDialog, QLabel, QFrame, QPushButton, QScrollArea,
    QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt6.QtCore import Qt


class QFullContent(QDialog):
    """
    Class to show the Full Content of a Document.
    """

    def __init__(
            self, doc_id: str, full_content: str, parent_widget: QWidget = None
    ):
        """
        Initialize class and attributes.
        """
        # Initialize the base class.
        super().__init__(parent_widget)
        self.setModal(False)

        # Save Class Attributes.
        self.doc_id = doc_id
        self.full_content = full_content

        # Create the UI.
        self.initializeUI()

    def initializeUI(self):
        """
        Set up the Application's GUI.
        """
        self.setMinimumSize(800, 600)
        self.setWindowTitle(f"Full Content - Doc<{self.doc_id}>")
        self.setUpMainWindow()

    def setUpMainWindow(self):
        """
        Create and arrange widgets in the main window.
        """
        # Create Text being shown.
        header_text = f"""
            <p align='left'><font face='Times New Roman' size='+1'><u>
            Full Content of Document &lt;{self.doc_id}&gt;
            </u></font></p>
        """
        content_text = self.full_content

        # Name Label.
        name_label = QLabel(header_text)
        name_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        name_layout = QHBoxLayout()
        name_layout.addSpacing(5)
        name_layout.addWidget(name_label, 0, Qt.AlignmentFlag.AlignLeft)

        # Content Area.
        text_read_only = QLabel(content_text)
        text_read_only.setWordWrap(True)
        text_read_only.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        text_read_layout = QVBoxLayout()
        text_read_layout.addWidget(text_read_only)
        text_read_layout.addStretch()
        text_read_container = QFrame()
        text_read_container.setFrameStyle(
            QFrame.Shape.Box | QFrame.Shadow.Sunken
        )
        text_read_container.setLayout(text_read_layout)
        # -- Text Container - Scrollable --
        text_read_scroll = QScrollArea()
        text_read_scroll.setWidget(text_read_container)
        text_read_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        text_read_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        text_read_scroll.setWidgetResizable(True)

        # Done Button.
        done_button = QPushButton('Done')
        # noinspection PyTypeChecker
        done_button.clicked.connect(self.close)

        # Create the Main Layout.
        main_v_box = QVBoxLayout()
        main_v_box.addLayout(name_layout)
        main_v_box.addWidget(text_read_scroll)
        main_v_box.addWidget(done_button, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(main_v_box)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QFullContent(
        doc_id='szt89e2',
        full_content='History of a song that was never sung before, but it was '
                     'very good. I really recommend this story for all the people '
                     'that has never heard this type of history before.'
    )
    window.show()
    sys.exit(app.exec())
