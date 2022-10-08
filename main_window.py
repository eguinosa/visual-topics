# Gelin Eguinosa Rosique
# 2022

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QTabWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from ir_system import IRSystem
from extra_funcs import progress_msg


class MainWindow(QMainWindow):
    """
    Main window of the IR system application.
    """
    # Supported Topic Models.
    supported_models = [
        'sbert_fast_500_docs_7_topics',
        'specter_sbert_fast_1_000_docs_9_topics',
        'sbert_fast_5_000_docs_55_topics',
        'specter_sbert_fast_5_000_docs_25_topics',
    ]

    def __init__(self, default_model='', show_progress=False):
        """
        Create the basic attributes to build the user interface.

        Args:
            default_model: String with the name of the default model for the
                system.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Initialize Base Class.
        super().__init__()

        # Check the default model used to start the app.
        if default_model:
            # Check we support the given model.
            if default_model in self.supported_models:
                current_model = default_model
            else:
                raise NameError(f"The model <{default_model} is not supported.")
        else:
            # Use the first of the supported models.
            current_model = self.supported_models[0]
        # Create the IR system.
        if show_progress:
            progress_msg("Creating the IR system for the application...")
        search_engine = IRSystem(model_name=current_model, show_progress=show_progress)

        # Save Class Attributes.
        self.current_model = current_model
        self.search_engine = search_engine

        # Create App UI.
        self.initializeUI(show_progress=show_progress)

    def initializeUI(self, show_progress=False):
        """
        Set up the Application's GUI
        """
        # Create Window Size and Name.
        self.setMinimumSize(1000, 800)
        self.setWindowTitle("Scientific Search")

        # Organize Layout.
        self.setUpMainWindow(show_progress=show_progress)
        self.createActions(show_progress=show_progress)
        self.createMenu(show_progress=show_progress)

    def setUpMainWindow(self, show_progress=False):
        """
        Create and arrange widgets in the main window.
        """
        # Get the Index of the current Topic Model.
        current_index = self.supported_models.index(self.current_model)

        # Topic Model - Label.
        model_label = QLabel("Topic Model:")
        # Topic Model - Combo Box.
        model_combo = QComboBox()
        model_combo.addItems(self.supported_models)
        model_combo.setCurrentIndex(current_index)
        model_combo.activated.connect(
            lambda index: self.switchModel(index, show_progress=show_progress)
        )

        # Create Search, Topics & Documents Layout Widgets.
        search_tab = self.createSearchTab(show_progress=show_progress)
        topics_tab = self.createTopicsTab(show_progress=show_progress)
        docs_tab = self.createDocsTab(show_progress=show_progress)
        # Create Tab.
        tab_bar = QTabWidget()
        tab_bar.addTab(search_tab, "Search")
        tab_bar.addTab(topics_tab, "Topics")
        tab_bar.addTab(docs_tab, "Document")

        # Topic Model - Layout.
        model_h_box = QHBoxLayout()
        model_h_box.addWidget(model_label)
        model_h_box.addWidget(model_combo)
        model_h_box.addStretch()
        model_widget = QWidget()
        model_widget.setLayout(model_h_box)

        # Central Widget - Create Main Layout.
        main_v_box = QVBoxLayout()
        main_v_box.addWidget(model_widget, 0, Qt.AlignmentFlag.AlignTop)
        main_v_box.addWidget(tab_bar)
        container = QWidget()
        container.setLayout(main_v_box)
        self.setCentralWidget(container)

    def createActions(self, show_progress=False):
        """
        Create the application's menu actions.
        """
        # Create actions for File menu.
        self.quit_act = QAction("&Quit")
        self.quit_act.setShortcut("Ctr+Q")
        self.quit_act.triggered.connect(self.close)

    def createMenu(self, show_progress=False):
        """
        Create the application's menu bar.
        """
        self.menuBar().setNativeMenuBar(False)
        # Create File Menu and actions.
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.quit_act)

    def switchModel(self, new_index, show_progress=False):
        """
        The ComboBox to manage the Topic Models has been activated, change the
        current model.
        """
        # Check if we have a new model name.
        new_model_name = self.supported_models[new_index]
        if new_model_name != self.current_model:
            if show_progress:
                progress_msg("Changing Topic Model of the IR system...")
            # Update the Topic Model of the IR System.
            self.search_engine.update_model(
                new_model=new_model_name, show_progress=show_progress
            )
            # Update the Name of the Current Topic Model.
            self.current_model = new_model_name
            if show_progress:
                progress_msg("Topic Model Updated!")
        elif show_progress:
            progress_msg("No need to update the Model.")

    def createSearchTab(self, show_progress=False):
        """
        Create the Widget with the layout in the Search Tab.
        """
        if show_progress:
            progress_msg("Creating Search Tab...")
        return QWidget()

    def createTopicsTab(self, show_progress=False):
        """
        Create the Widget with the layout in the Topics Tab.
        """
        if show_progress:
            progress_msg("Creating Topics Tab...")
        return QWidget()

    def createDocsTab(self, show_progress=False):
        """
        Create the Widget with the layout in the Documents Tab.
        """
        if show_progress:
            progress_msg("Creating Documents Tab...")
        return QWidget()


# Run Application.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(show_progress=True)
    window.show()
    sys.exit(app.exec())
