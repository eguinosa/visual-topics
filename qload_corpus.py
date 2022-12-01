# Gelin Eguinosa Rosique
# 2022

import sys
from PyQt6.QtWidgets import (
    QApplication, QDialog, QWidget, QLabel, QRadioButton, QButtonGroup,
    QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QComboBox, QFileDialog
)
from PyQt6.QtCore import Qt


class QLoadCorpusDialog(QDialog):
    """
    Class to create a Topic Model for the IR System with a new corpus.
    """

    def __init__(
            self, sbert_names: list, specter_names: list,
            parent_widget: QWidget = None
    ):
        """
        Initialize class and attributes.
        """
        # Initialize the base class.
        super().__init__(parent_widget)
        self.setModal(True)

        # Save class attributes.
        self.sbert_names = sbert_names
        self.specter_names = specter_names

        # Dialog Widgets.
        self.topic_group = None
        self.path_line_edit = None
        self.sbert_combo = None
        self.specter_combo = None

        # New Model Info.
        self.new_corpus_path = ''

        # Create the UI.
        self.initializeUI()

    def initializeUI(self):
        """
        Set up the Application's GUI.
        """
        self.setMinimumSize(500, 350)
        self.setWindowTitle("Load New Corpus")
        self.setUpMainWindow()

    def setUpMainWindow(self):
        """
        Create and arrange widgets in the main window.
        """
        # - Topic Model Area -
        topic_model_label = QLabel("Topic Model:")
        # Topic Model's Radio Buttons.
        sbert_rb = QRadioButton('SBERT')
        sbert_rb.setChecked(True)
        specter_rb = QRadioButton('SPECTER-SBERT')
        # Add Radio Buttons to group.
        topic_group = QButtonGroup()
        self.topic_group = topic_group
        topic_group.addButton(sbert_rb)
        topic_group.addButton(specter_rb)
        topic_group.buttonToggled.connect(
            lambda button, checked: self.checkModelType()
        )
        # Radio Button Layout.
        topic_layout = QHBoxLayout()
        topic_layout.addWidget(sbert_rb)
        topic_layout.addWidget(specter_rb)
        topic_layout.addStretch()

        # - Corpus Folder Area -
        corpus_folder_label = QLabel("Corpus Folder:")
        path_line_edit = QLineEdit()
        self.path_line_edit = path_line_edit
        path_line_edit.setReadOnly(True)
        path_line_edit.setPlaceholderText("Open Folder...")
        open_folder_button = QPushButton("Open")
        open_folder_button.setAutoDefault(False)
        # noinspection PyUnresolvedReferences
        open_folder_button.clicked.connect(
            lambda checked: self.openCorpusFolder()
        )
        # Folder Layout.
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(path_line_edit)
        folder_layout.addWidget(open_folder_button)

        # - Models Area -
        # SBERT Model
        sbert_model_label = QLabel("SBERT Model:")
        sbert_combo = QComboBox()
        self.sbert_combo = sbert_combo
        sbert_combo.addItems(self.sbert_names)
        # SPECTER-SBERT Model.
        specter_model_label = QLabel("SPECTER Model:")
        specter_combo = QComboBox()
        self.specter_combo = specter_combo
        specter_combo.addItems(self.specter_names)
        specter_combo.setEnabled(False)

        # Button - Load Corpus & Create New Topic Model.
        load_button = QPushButton("Load")
        load_button.setDefault(True)
        # noinspection PyUnresolvedReferences
        load_button.clicked.connect(
            lambda checked: self.createNewTopicModel()
        )

        # - Main Layout -
        main_v_box = QVBoxLayout()
        main_v_box.addWidget(topic_model_label, 0, Qt.AlignmentFlag.AlignLeft)
        main_v_box.addLayout(topic_layout)
        main_v_box.addSpacing(10)
        main_v_box.addStretch()
        main_v_box.addWidget(corpus_folder_label, 0, Qt.AlignmentFlag.AlignLeft)
        main_v_box.addLayout(folder_layout)
        main_v_box.addWidget(sbert_model_label, 0, Qt.AlignmentFlag.AlignLeft)
        main_v_box.addWidget(sbert_combo, 0, Qt.AlignmentFlag.AlignLeft)
        main_v_box.addWidget(specter_model_label, 0, Qt.AlignmentFlag.AlignLeft)
        main_v_box.addWidget(specter_combo, 0, Qt.AlignmentFlag.AlignLeft)
        main_v_box.addStretch()
        main_v_box.addWidget(load_button, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(main_v_box)

    def openCorpusFolder(self):
        """
        Open the Folder containing the '.txt' files with the documents in the
        corpus.
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open Corpus Folder", "",
            QFileDialog.Option.ShowDirsOnly |
            QFileDialog.Option.DontResolveSymlinks
        )
        if folder_path:
            # Save path to the new corpus.
            self.new_corpus_path = folder_path
            # Show the Path on the Line Edit.
            self.path_line_edit.setText(folder_path)

    def checkModelType(self):
        """
        Check to selected Model Type to enable or disable the SPECTER Model
        Combo Box.
        """
        sel_model_type = self.topic_group.checkedButton().text().lower()
        if sel_model_type == 'specter-sbert':
            self.specter_combo.setEnabled(True)
        else:
            self.specter_combo.setEnabled(False)

    def createNewTopicModel(self):
        """
        Create a new Topic Model using the new loaded Corpus.
        """
        if self.new_corpus_path:
            # Get Data of the New Topic Model.
            topic_model_type = self.topic_group.checkedButton().text().lower()
            new_corpus_path = self.new_corpus_path
            sbert_model_name = self.sbert_combo.currentText()
            specter_model_name = self.specter_combo.currentText()
            # Test - Display the Data we are going to use to build the new model.
            if topic_model_type == 'specter-sbert':
                print(f"Model Type: {topic_model_type}")
                print(f"Corpus Path: {new_corpus_path}")
                print(f"SPECTER Model: {specter_model_name}")
                print(f"SBERT Model: {sbert_model_name}")
            else:
                print(f"Model Type: {topic_model_type}")
                print(f"Corpus Path: {new_corpus_path}")
                print(f"SBERT Model: {sbert_model_name}")
            # Accepted.
            self.accept()
        else:
            # No Action can be taken without a new Corpus.
            print("No Corpus provided!")
            self.reject()

    def collectedInfoDict(self):
        """
        Create a Dictionary with the Info about the new Corpus and the language
        models of the Topic Model.
        """
        # Corpus & Topic Model Data.
        topic_model_type = self.topic_group.checkedButton().text().lower()
        new_corpus_path = self.new_corpus_path
        sbert_model_name = self.sbert_combo.currentText()
        specter_model_name = self.specter_combo.currentText()
        # Create Dict.
        new_data_dict = {
            'corpus_path': new_corpus_path,
            'topic_model': topic_model_type,
            'sbert_model': sbert_model_name,
            'specter_model': specter_model_name,
        }
        # Dictionary with the Info about Corpus & Topic Model.
        return new_data_dict


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # noinspection PyTypeChecker
    window = QLoadCorpusDialog(
        sbert_names=['sbert_fast', 'sbert_best', 'sbert_multilingual'],
        specter_names=['specter_default', 'specter_new']
    )
    window.show()
    sys.exit(app.exec())
