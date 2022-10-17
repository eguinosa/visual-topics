# Gelin Eguinosa Rosique
# 2022

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QTabWidget, QLayout, QScrollArea, QLineEdit, QPushButton, QFrame,
    QButtonGroup, QCheckBox, QRadioButton, QDialog, QProgressDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from ir_system import IRSystem
from qmodel_worker import QModelWorker
from qupdates_dialog import QUpdatesDialog
from qvocab_dialog import QVocabDialog
from qfull_content import QFullContent
from qopen_document import QOpenDocument
from extra_funcs import progress_msg


class MainWindow(QMainWindow):
    """
    Main window of the IR system application.
    """
    # Supported Topic Models.
    supported_models = [
        'specter_sbert_fast_5_000_docs_25_topics',
        'specter_sbert_fast_20_000_docs_119_topics',
        'specter_sbert_fast_105_548_docs_533_topics',
        'sbert_fast_5_000_docs_55_topics',
        'sbert_fast_20_000_docs_182_topics',
        'sbert_fast_105_548_docs_745_topics',
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

        # -- Load the Topic Model --
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

        # -- Create IR System with Topic Model --
        if show_progress:
            progress_msg("Creating the IR system for the application...")
        search_engine = IRSystem(model_name=current_model, show_progress=show_progress)

        # Save the Data Related with the Current Topic Model.
        topic_size = str(search_engine.topic_size)
        supported_sizes = [str(x) for x in search_engine.supported_model_sizes()]
        cur_size_index = supported_sizes.index(topic_size)

        # Save Class Attributes.
        self.current_model = current_model
        self.search_engine = search_engine
        # Topic Size Attributes.
        self.topic_size = topic_size
        self.supported_sizes = supported_sizes
        self.cur_size_index = cur_size_index

        # Search Tab - Fixed Values
        self.search_tab_index = 0
        self.search_tab_docs_num = 15
        self.search_tab_topics_num = 10
        self.search_tab_word_num = 10
        self.search_tab_preview_size = 300
        self.search_tab_init_query = 'covid-19 pandemic'
        # Search Tab - Variables.
        self.search_tab_working = False
        self.search_tab_query_text = ''
        self.search_tab_query_embed = None
        self.search_tab_top_topic = ''
        self.search_tab_all_topics = False
        self.search_tab_selected_topics = None
        self.search_tab_topics_info = None
        self.search_tab_docs_info = None
        self.search_tab_topic_checkboxes = None
        # Search Tab - Info Widgets
        self.search_tab_size_combo = None
        self.search_tab_all_checkbox = None
        self.search_tab_line_edit = None
        self.search_tab_docs_scroll = None
        self.search_tab_topics_scroll = None

        # Topics Tab - Fixed Values
        self.topics_tab_index = 1
        self.topics_tab_word_num = 15
        self.topics_tab_preview_size = 200
        self.topics_tab_sort_cats = ['Size', 'PWI-tf-idf', 'PWI-exact']
        self.topics_tab_topics_page_size = 20
        self.topics_tab_docs_page_size = 50
        # Topics Tab - Topic Variables
        self.topics_tab_cur_cat = ''
        self.topics_tab_cat_index = 0
        self.topics_tab_cur_topic = ''
        self.topics_tab_topics_cur_page = ''
        self.topics_tab_topics_total_pages = ''
        self.topics_tab_topics_pages = None
        self.topics_tab_topics_page_info = None
        self.topics_tab_button_group = None
        self.topics_tab_topics_radios = None
        self.topics_tab_topics_widgets = None
        # Topic Tab - Doc Variables
        self.topics_tab_docs_num = 0
        self.topics_tab_docs_cur_page = ''
        self.topics_tab_docs_total_pages = ''
        self.topics_tab_docs_page_info = None
        # Topics Tab - Info Widgets
        self.topics_tab_size_combo = None
        self.topics_tab_topics_scroll = None
        self.topics_tab_docs_label = None
        self.topics_tab_docs_scroll = None
        self.topics_tab_topics_first_button = None
        self.topics_tab_topics_prev_button = None
        self.topics_tab_topics_next_button = None
        self.topics_tab_topics_last_button = None
        self.topics_tab_topics_page_label = None
        self.topics_tab_docs_first_button = None
        self.topics_tab_docs_prev_button = None
        self.topics_tab_docs_next_button = None
        self.topics_tab_docs_last_button = None
        self.topics_tab_docs_page_label = None

        # Documents Tab - Fixed Values
        self.docs_tab_index = 2
        self.docs_tab_topics_num = 10
        self.docs_tab_docs_num = 10
        self.docs_tab_word_num = 10
        self.docs_tab_preview_size = 200
        # Documents Tab - Variables
        self.docs_tab_working = False
        self.docs_tab_cur_doc = ''
        self.docs_tab_doc_embed = None
        self.docs_tab_doc_content = ''
        self.docs_tab_top_topic = ''
        self.docs_tab_all_topics = False
        self.docs_tab_selected_topics = None
        self.docs_tab_topics_info = None
        self.docs_tab_docs_info = None
        self.docs_tab_topic_checkboxes = None
        # Documents Tab - Info Widgets
        self.docs_tab_size_combo = None
        self.docs_tab_all_checkbox = None
        self.docs_tab_topics_scroll = None
        self.docs_tab_name_label = None
        self.docs_tab_content_button = None
        self.docs_tab_text_widget = None
        self.docs_tab_docs_scroll = None

        # Main Window - Information Values & Widgets
        self.current_tab_index = self.topics_tab_index
        self.main_tab_bar = None
        self.size_combo_working = False

        # Menu Bar Actions.
        self.quit_act = None
        self.open_doc = None
        self.random_doc = None

        # Dialog Windows & Thread Workers.
        self.model_worker = None
        self.update_dialog = None
        self.vocab_window = None
        self.content_window = None

        # Create the Values for the Search Tab.
        self.updateSearchTabVariables(query_text=self.search_tab_init_query)
        # Create the Values for the Topics Tab.
        self.updateTopicsTabTopicVariables(cur_cat_index=0)
        self.updateTopicsTabDocVariables()
        # Create the Values for the Documents Tab.
        cur_doc_id = search_engine.random_doc_id()
        self.updateDocsTabVariables(cur_doc_id=cur_doc_id)
        # Create App UI.
        self.initializeUI(show_progress=show_progress)

    def initializeUI(self, show_progress=False):
        """
        Set up the Application's GUI
        """
        # Create Window Size and Name.
        self.setMinimumSize(1000, 800)
        self.setGeometry(10, 80, 1280, 800)
        self.setWindowTitle("Scientific Search")
        # Change the Location of the Window.
        # self.move(10, 80)
        # self.resize(1280, 800)

        # Organize Layout.
        self.setUpMainWindow(show_progress=show_progress)
        self.createActions(show_progress=show_progress)
        self.createMenu(show_progress=show_progress)
        # # Check Window Size Pre-Showing.
        # progress_msg(f"Self -> Window Width: {self.width()}")
        # progress_msg(f"Self -> Window Height: {self.height()}")

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
        self.main_tab_bar = tab_bar
        tab_bar.addTab(search_tab, "Search")
        tab_bar.addTab(topics_tab, "Topics")
        tab_bar.addTab(docs_tab, "Documents")
        tab_bar.setCurrentIndex(self.current_tab_index)
        tab_bar.currentChanged.connect(
            lambda index: self.newTabActivated(new_index=index)
        )

        # Topic Model - Layout.
        model_h_box = QHBoxLayout()
        model_h_box.addWidget(model_label)
        model_h_box.addWidget(model_combo)
        model_h_box.addStretch()
        model_container = QWidget()
        model_container.setLayout(model_h_box)

        # Central Widget - Create Main Layout.
        main_v_box = QVBoxLayout()
        main_v_box.addWidget(model_container, 0, Qt.AlignmentFlag.AlignTop)
        main_v_box.addWidget(tab_bar)
        main_container = QWidget()
        main_container.setLayout(main_v_box)
        self.setCentralWidget(main_container)

    def createActions(self, show_progress=False):
        """
        Create the application's menu actions.
        """
        # Create actions for File menu.
        self.quit_act = QAction("&Quit")
        self.quit_act.setShortcut("Ctrl+Q")
        # noinspection PyTypeChecker
        self.quit_act.triggered.connect(self.close)
        # Create Actions for Documents Tab.
        self.open_doc = QAction("Open Document")
        self.open_doc.setShortcut("Ctrl+O")
        self.open_doc.triggered.connect(self.getDocID)
        self.random_doc = QAction("Random Document")
        self.random_doc.setShortcut("Ctrl+R")
        self.random_doc.triggered.connect(self.openRandomDoc)

        if show_progress:
            progress_msg("Actions Created!")

    def createMenu(self, show_progress=False):
        """
        Create the application's menu bar.
        """
        self.menuBar().setNativeMenuBar(False)
        # Create File Menu and actions.
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.quit_act)
        # Create Documents Tab Menu.
        doc_menu = self.menuBar().addMenu("Docs")
        doc_menu.addAction(self.open_doc)
        doc_menu.addAction(self.random_doc)
        if show_progress:
            progress_msg("Menu Created!")

    def createSearchTab(self, show_progress=False):
        """
        Create the Widget with the layout in the Search Tab.
        """
        # --- Search & Documents Area ---
        # - Search Section -
        search_line_edit = QLineEdit()
        self.search_tab_line_edit = search_line_edit
        search_line_edit.setPlaceholderText(f" {self.search_tab_query_text}")
        search_line_edit.returnPressed.connect(
            lambda: self.searchRequested()
        )
        search_button = QPushButton("Search")
        search_button.clicked.connect(
            lambda checked: self.searchRequested()
        )
        # Search Layout
        search_layout = QHBoxLayout()
        search_layout.addWidget(search_line_edit)
        search_layout.addWidget(search_button, 0, Qt.AlignmentFlag.AlignRight)
        # - Top Documents Section -
        top_docs_num = self.search_tab_docs_num
        top_docs_label = QLabel(f"Top {top_docs_num} Documents:")
        docs_scroll_area = QScrollArea()
        self.search_tab_docs_scroll = docs_scroll_area
        docs_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        docs_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        docs_scroll_area.setWidgetResizable(True)
        # Update Content in the Scrollable.
        self.updateSearchTabDocsScroll()
        # -- Search & Documents Layout --
        search_docs_layout = QVBoxLayout()
        search_docs_layout.addLayout(search_layout)
        search_docs_layout.addWidget(top_docs_label, 0, Qt.AlignmentFlag.AlignLeft)
        search_docs_layout.addWidget(docs_scroll_area)
        search_docs_container = QWidget()
        search_docs_container.setLayout(search_docs_layout)

        # --- Topic Area ---
        # - Topic Size Section -
        size_label = QLabel("Topic Size:")
        size_combo = QComboBox()
        self.search_tab_size_combo = size_combo
        size_combo.addItems(self.supported_sizes)
        size_combo.setCurrentIndex(self.cur_size_index)
        size_combo.activated.connect(
            lambda index: self.changeTopicSize(index, show_progress=show_progress)
        )
        # Size Layout.
        size_layout = QHBoxLayout()
        size_layout.addWidget(size_label)
        size_layout.addWidget(size_combo)
        size_layout.addStretch()
        size_container = QWidget()
        size_container.setLayout(size_layout)
        # - All Topics Checkbox -
        all_topics_checkbox = QCheckBox("Use All Topics")
        self.search_tab_all_checkbox = all_topics_checkbox
        all_topics_checkbox.toggled.connect(
            lambda checked: self.newSearchTabAllTopics(checked=checked)
        )
        # - Top Topics Area -
        top_topics_num = self.search_tab_topics_num
        top_topics_label = QLabel(f"Top {top_topics_num} Topics:")
        topics_scroll_area = QScrollArea()
        self.search_tab_topics_scroll = topics_scroll_area
        topics_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        topics_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        topics_scroll_area.setWidgetResizable(True)
        # Update Content of the Topics Scrollable.
        self.updateSearchTabTopicsScroll()
        # -- Topics Layout --
        topic_area_layout = QVBoxLayout()
        topic_area_layout.addWidget(size_container, 0, Qt.AlignmentFlag.AlignRight)
        topic_area_layout.addWidget(all_topics_checkbox, 0, Qt.AlignmentFlag.AlignLeft)
        topic_area_layout.addWidget(top_topics_label, 0, Qt.AlignmentFlag.AlignLeft)
        topic_area_layout.addWidget(topics_scroll_area)
        topic_area_container = QWidget()
        topic_area_container.setLayout(topic_area_layout)

        # --- Search Tab Layout ---
        search_tab_layout = QHBoxLayout()
        search_tab_layout.addWidget(search_docs_container, 2)
        search_tab_layout.addWidget(topic_area_container, 1)
        search_tab_container = QWidget()
        search_tab_container.setLayout(search_tab_layout)
        # The Search Tab Completely Built.
        if show_progress:
            progress_msg("Search Tab Complete!")
        return search_tab_container

    def createTopicsTab(self, show_progress=False):
        """
        Create the Widget with the layout in the Topics Tab.
        """
        # --- All the Topic Area ---
        # - Topic's Size Area -
        size_label = QLabel("Topic Size:")
        size_combo = QComboBox()
        self.topics_tab_size_combo = size_combo
        size_combo.addItems(self.supported_sizes)
        size_combo.setCurrentIndex(self.cur_size_index)
        size_combo.activated.connect(
            lambda index: self.changeTopicSize(index, show_progress=show_progress)
        )
        # Topic's Size Layout.
        size_layout = QHBoxLayout()
        size_layout.addWidget(size_label)
        size_layout.addWidget(size_combo)
        size_layout.addStretch()
        # - Top Topics Area -
        # Category used to sort the Topics.
        sort_cat_label = QLabel("Topics by:")
        sort_cats_combo = QComboBox()
        sort_cats_combo.addItems(self.topics_tab_sort_cats)
        sort_cats_combo.setCurrentIndex(self.topics_tab_cat_index)
        sort_cats_combo.activated.connect(
            lambda index: self.changeTopicSorting(index, show_progress=show_progress)
        )
        sort_cats_layout = QHBoxLayout()
        sort_cats_layout.addWidget(sort_cat_label)
        sort_cats_layout.addWidget(sort_cats_combo)
        sort_cats_layout.addStretch()
        # Create Scrollable Area with the Topics.
        topics_scroll_area = QScrollArea()
        self.topics_tab_topics_scroll = topics_scroll_area
        topics_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        topics_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        topics_scroll_area.setWidgetResizable(True)
        # - Topics Page Area -
        topic_first_button = QPushButton("<<")
        self.topics_tab_topics_first_button = topic_first_button
        topic_first_button.clicked.connect(
            lambda checked: self.firstPageTopicsTabTopicScroll()
        )
        topic_prev_button = QPushButton("Prev")
        self.topics_tab_topics_prev_button = topic_prev_button
        topic_prev_button.clicked.connect(
            lambda checked: self.prevPageTopicsTabTopicScroll()
        )
        topic_next_button = QPushButton("Next")
        self.topics_tab_topics_next_button = topic_next_button
        topic_next_button.clicked.connect(
            lambda checked: self.nextPageTopicsTabTopicScroll()
        )
        topic_last_button = QPushButton(">>")
        self.topics_tab_topics_last_button = topic_last_button
        topic_last_button.clicked.connect(
            lambda checked: self.lastPageTopicsTabTopicsScroll()
        )
        topic_page_label = QLabel()
        self.topics_tab_topics_page_label = topic_page_label
        # Page Layout.
        topic_page_layout = QHBoxLayout()
        topic_page_layout.setSpacing(4)
        topic_page_layout.addSpacing(10)
        topic_page_layout.addWidget(topic_first_button)
        topic_page_layout.addWidget(topic_prev_button)
        topic_page_layout.addWidget(topic_next_button)
        topic_page_layout.addWidget(topic_last_button)
        topic_page_layout.addStretch()
        topic_page_layout.addWidget(topic_page_label)
        topic_page_layout.addSpacing(10)
        # -- All the Topic Area Layout --
        topic_area_layout = QVBoxLayout()
        topic_area_layout.addLayout(size_layout)
        topic_area_layout.addLayout(sort_cats_layout)
        topic_area_layout.addWidget(topics_scroll_area)
        topic_area_layout.addLayout(topic_page_layout)
        topic_area_container = QWidget()
        topic_area_container.setLayout(topic_area_layout)
        # -- Update Topics Scrollable --
        self.updateTopicsTabTopicScroll()

        # --- Top Docs Area ---
        # Create Topic Documents Label.
        self.topics_tab_docs_label = QLabel()
        # Create Scrollable Area for the Documents.
        docs_scroll_area = QScrollArea()
        self.topics_tab_docs_scroll = docs_scroll_area
        docs_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        docs_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        docs_scroll_area.setWidgetResizable(True)
        # - Documents Page Area -
        doc_first_button = QPushButton("<<")
        self.topics_tab_docs_first_button = doc_first_button
        doc_first_button.clicked.connect(
            lambda checked: self.firstPageTopicsTabDocScroll()
        )
        doc_prev_button = QPushButton("Prev")
        self.topics_tab_docs_prev_button = doc_prev_button
        doc_prev_button.clicked.connect(
            lambda checked: self.prevPageTopicsTabDocScroll()
        )
        doc_next_button = QPushButton("Next")
        self.topics_tab_docs_next_button = doc_next_button
        doc_next_button.clicked.connect(
            lambda checked: self.nextPageTopicsTabDocScroll()
        )
        doc_last_button = QPushButton(">>")
        self.topics_tab_docs_last_button = doc_last_button
        doc_last_button.clicked.connect(
            lambda checked: self.lastPageTopicsTabDocScroll()
        )
        doc_page_label = QLabel()
        self.topics_tab_docs_page_label = doc_page_label
        # Doc Page Layout.
        doc_page_layout = QHBoxLayout()
        doc_page_layout.setSpacing(4)
        doc_page_layout.addSpacing(10)
        doc_page_layout.addWidget(doc_first_button)
        doc_page_layout.addWidget(doc_prev_button)
        doc_page_layout.addWidget(doc_next_button)
        doc_page_layout.addWidget(doc_last_button)
        doc_page_layout.addStretch()
        doc_page_layout.addWidget(doc_page_label)
        doc_page_layout.addSpacing(10)
        # -- Top Docs Area Final Layout --
        docs_area_layout = QVBoxLayout()
        docs_area_layout.addWidget(self.topics_tab_docs_label)
        docs_area_layout.addWidget(self.topics_tab_docs_scroll)
        docs_area_layout.addLayout(doc_page_layout)
        docs_area_container = QWidget()
        docs_area_container.setLayout(docs_area_layout)
        # -- Update Documents Scrollable --
        self.updateTopicsTabDocScroll()

        # --- Topics Tab Layout ---
        topics_tab_layout = QHBoxLayout()
        topics_tab_layout.addWidget(topic_area_container)
        topics_tab_layout.addWidget(docs_area_container)
        topics_tab_container = QWidget()
        topics_tab_container.setLayout(topics_tab_layout)
        # The Topics Tab is Completely Built.
        if show_progress:
            progress_msg("Topics Tab Complete!")
        return topics_tab_container

    def createDocsTab(self, show_progress=False):
        """
        Create the Widget with the layout in the Documents Tab.
        """
        # --- Topic Area ---
        # Topic Size Area
        size_label = QLabel("Topic Size:")
        size_combo = QComboBox()
        self.docs_tab_size_combo = size_combo
        size_combo.addItems(self.supported_sizes)
        size_combo.setCurrentIndex(self.cur_size_index)
        size_combo.activated.connect(
            lambda index: self.changeTopicSize(index, show_progress=show_progress)
        )
        # - Topic's Size Layout -
        size_layout = QHBoxLayout()
        size_layout.addWidget(size_label)
        size_layout.addWidget(size_combo)
        size_layout.addStretch()
        # - All Topics Checkbox -
        all_topics_checkbox = QCheckBox("Use All Topics")
        self.docs_tab_all_checkbox = all_topics_checkbox
        all_topics_checkbox.toggled.connect(
            lambda checked:
            self.newDocsTabAllTopics(checked=checked, show_progress=show_progress)
        )
        # - Top Topics Area -
        top_topics_num = self.docs_tab_topics_num
        close_topics_label = QLabel(f"Top {top_topics_num} Closest Topics:")
        topics_scroll_area = QScrollArea()
        self.docs_tab_topics_scroll = topics_scroll_area
        topics_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        topics_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        topics_scroll_area.setWidgetResizable(True)
        # Update Content of Scrollable.
        self.updateDocsTabTopicsScroll()
        # -- Save the Topic Area Layout --
        topic_area_layout = QVBoxLayout()
        topic_area_layout.addLayout(size_layout)
        topic_area_layout.addWidget(all_topics_checkbox, 0, Qt.AlignmentFlag.AlignLeft)
        topic_area_layout.addWidget(close_topics_label, 0, Qt.AlignmentFlag.AlignLeft)
        topic_area_layout.addWidget(topics_scroll_area)
        topic_area_container = QWidget()
        topic_area_container.setLayout(topic_area_layout)

        # --- Doc Content Area ---
        doc_id = self.docs_tab_cur_doc
        has_full_content = self.docs_tab_doc_content != ''
        # Header Area.
        name_label = QLabel()
        self.docs_tab_name_label = name_label
        name_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        full_content_button = QPushButton("Full Content")
        self.docs_tab_content_button = full_content_button
        full_content_button.setEnabled(has_full_content)
        full_content_button.clicked.connect(
            lambda checked, x=doc_id: self.viewFullContent(doc_id=x)
        )
        # - Header Area Layout -
        header_layout = QHBoxLayout()
        header_layout.addWidget(name_label, 0, Qt.AlignmentFlag.AlignLeft)
        header_layout.addWidget(full_content_button, 0, Qt.AlignmentFlag.AlignRight)
        # Content Text Area.
        text_read_only = QLabel()
        self.docs_tab_text_widget = text_read_only
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
        text_read_scroll = QScrollArea()
        text_read_scroll.setWidget(text_read_container)
        text_read_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        text_read_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        text_read_scroll.setWidgetResizable(True)
        # Update the Info in the Content Area.
        self.updateDocsTabContentArea()
        # -- Doc Content Area Layout --
        doc_content_layout = QVBoxLayout()
        doc_content_layout.addLayout(header_layout)
        doc_content_layout.addWidget(text_read_scroll)
        doc_content_container = QWidget()
        doc_content_container.setLayout(doc_content_layout)

        # --- Close Docs Area ---
        top_docs_num = self.docs_tab_docs_num
        close_docs_label = QLabel(f"Top {top_docs_num} Close Documents:")
        docs_scroll_area = QScrollArea()
        self.docs_tab_docs_scroll = docs_scroll_area
        docs_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        docs_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        docs_scroll_area.setWidgetResizable(True)
        # Update content of Scrollable.
        self.updateDocsTabDocsScroll()
        # -- Close Docs Area Layout --
        close_docs_layout = QVBoxLayout()
        close_docs_layout.addWidget(close_docs_label)
        close_docs_layout.addWidget(docs_scroll_area)
        close_docs_container = QWidget()
        close_docs_container.setLayout(close_docs_layout)

        # --- Documents Tab Layout ---
        docs_tab_layout = QHBoxLayout()
        docs_tab_layout.addWidget(topic_area_container, 27)
        docs_tab_layout.addWidget(doc_content_container, 46)
        docs_tab_layout.addWidget(close_docs_container, 27)
        docs_tab_container = QWidget()
        docs_tab_container.setLayout(docs_tab_layout)
        # The Documents Tab is complete.
        if show_progress:
            progress_msg("Documents Tab Complete!")
        return docs_tab_container

    def createTopicItem(
            self, topic_id: str, cat_type: str, cat_value, description='',
            view_type='topic', checkable_type='search-checkbox', set_checked=False
    ):
        """
        Build a List Item for the given 'topic_id'. The Item built depends on
        the information provided, if description is empty, the use the search
        engine to load it.

        The variable 'checkable_type' determines the type of check button will
        be used for the topics, either QCheckboxes (search-checkbox,
        doc-checkbox) or QRadioButton (radio-button).
        """
        # Create Layout from Start.
        topic_item_layout = QVBoxLayout()

        # Check we have a description.
        if not description:
            description = self.search_engine.topic_description(topic_id=topic_id)
        topic_descript = f"""<p>{description}</p>"""
        # Create the Topic Header String.
        topic_header = topic_id
        if cat_type == 'similarity':
            topic_header += f" (Similarity: {round(cat_value, 3)})"
        elif cat_type == 'pwi-tf-idf':
            topic_header += f" (PWI-tf-idf: {round(cat_value, 4)})"
        elif cat_type == 'pwi-exact':
            topic_header += f" (PWI-exact: {round(cat_value, 4)})"
        elif cat_type == 'size':
            topic_header += f" (Size: {cat_value} docs)"

        # Create Header Checkbox or Radio Button.
        if checkable_type in {'search-checkbox', 'doc-checkbox'}:
            header_checkable = QCheckBox(topic_header)
            if set_checked:
                header_checkable.setChecked(True)
            # Set method called when we check the button.
            if checkable_type == 'search-checkbox':
                self.search_tab_topic_checkboxes[topic_id] = header_checkable
                header_checkable.toggled.connect(
                    lambda checked, x=topic_id:
                    self.newSearchTabTopicSelection(checked, topic_id=x)
                )
            elif checkable_type == 'doc-checkbox':
                self.docs_tab_topic_checkboxes[topic_id] = header_checkable
                header_checkable.toggled.connect(
                    lambda checked, x=topic_id:
                    self.newDocsTabTopicSelection(checked, topic_id=x)
                )
            else:
                raise NameError(f"Non-supported Checkable: {checkable_type}")
        elif checkable_type == 'radio-button':
            header_checkable = QRadioButton(topic_header)
            self.topics_tab_button_group.addButton(header_checkable)
            self.topics_tab_topics_radios[topic_id] = header_checkable
            if set_checked:
                header_checkable.setChecked(True)
            header_checkable.toggled.connect(
                lambda checked, x=topic_id: self.newTopicSelected(checked, x)
            )
        else:
            raise NameError(f"The Checkable Type <{checkable_type} is not supported.")
        # Add Header to Layout.
        topic_item_layout.addWidget(header_checkable)

        # Create the View Button.
        if view_type == 'topic':
            view_button = QPushButton("View Topic")
            view_button.clicked.connect(
                lambda checked, x=topic_id: self.viewTopic(topic_id=x)
            )
        elif view_type == 'vocabulary':
            view_button = QPushButton("Full Vocabulary")
            view_button.clicked.connect(
                lambda checked, x=topic_id: self.viewVocabulary(topic_id=x)
            )
        else:
            raise NameError(f"The View Type <{view_type}> is not supported.")
        # Add View Button to Layout.
        topic_item_layout.addWidget(view_button, 0, Qt.AlignmentFlag.AlignLeft)

        # Create the Description Label.
        descript_label = QLabel(topic_descript)
        descript_label.setWordWrap(True)
        descript_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        # Add Description Label to Layout.
        topic_item_layout.addWidget(descript_label)

        # Surround the Layout with a Frame.
        topic_item_container = QFrame()
        topic_item_container.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        topic_item_container.setLayout(topic_item_layout)
        # Item with the Info about the Topic.
        return topic_item_container

    def createDocItem(self, doc_id: str, title: str, abstract='', sim=0):
        """
        Build a List Item for the given 'doc_id'. The Returned Document Item
        will depend upon the provided Information, the empty fields will be
        omitted.
        """
        # Build Document Header Text.
        doc_header = f"Doc<{doc_id}>"
        if sim:
            doc_header += f" (Similarity: {round(sim, 3)})"
        # Build Document Content.
        if abstract:
            doc_content = f"""
                <p align='left'><font face='Times New Roman' size='+1'><u>
                {title}
                </u></font></p>
                <p align='left'>{abstract}</p>
            """
        else:
            doc_content = f"""
                <p align='left'><font face='Times New Roman' size='+2'>
                {title}
                </font></p>
            """
        # Create the Document Item Layout.
        header_label = QLabel(doc_header)
        header_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        view_button = QPushButton("View Document")
        view_button.clicked.connect(
            lambda checked, x=doc_id: self.viewDocument(doc_id=x)
        )
        content_label = QLabel(doc_content)
        content_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        content_label.setWordWrap(True)
        doc_item_layout = QVBoxLayout()
        doc_item_layout.addWidget(header_label)
        doc_item_layout.addWidget(view_button, 0, Qt.AlignmentFlag.AlignLeft)
        doc_item_layout.addWidget(content_label)
        doc_item_container = QFrame()
        doc_item_container.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        doc_item_container.setLayout(doc_item_layout)
        # The Item with the Doc Info.
        return doc_item_container

    def newTabActivated(self, new_index: int):
        """
        A New Tab in the Window has been activated, so update the value of the
        current Tab.
        """
        self.current_tab_index = new_index

    def switchModel(self, new_index, show_progress=False):
        """
        The ComboBox to manage the Topic Models has been activated, change the
        current model.
        """
        # Check if we have a new model name.
        new_model_name = self.supported_models[new_index]
        if new_model_name != self.current_model:
            # Update the Topic Model of the IR System.
            if show_progress:
                progress_msg("Changing Topic Model of the IR system...")

            # Use Threads to Change the Model.
            self.model_worker = QModelWorker(
                search_engine=self.search_engine, new_model_name=new_model_name,
                parent_widget=self, show_progress=show_progress
            )
            self.update_dialog = QUpdatesDialog(
                action_text="Updating Topic Model",
                message_text=f"Changing to the Topic Model <{new_model_name}>...",
                parent_widget=self
            )
            self.model_worker.task_done.connect(lambda: self.update_dialog.accept())
            # noinspection PyUnresolvedReferences
            self.model_worker.finished.connect(self.model_worker.deleteLater)

            # Start the thread and the Dialog.
            self.model_worker.start()
            if self.update_dialog.exec() == QDialog.DialogCode.Accepted:
                # Update the Name of the Current Topic Model.
                self.current_model = new_model_name
                # Update the Supported Sizes on the App.
                self.updateSupportedSizes()

                # Update Documents & Topics in the Search Tab.
                current_query = self.search_tab_query_text
                self.updateSearchTabVariables(
                    query_text=current_query, show_progress=show_progress
                )
                self.updateSearchTabTopicsScroll()
                self.updateSearchTabDocsScroll()
                # Update Topics & Documents in the Topics Tab.
                self.updateTopicsTabTopicVariables()
                self.updateTopicsTabDocVariables()
                self.updateTopicsTabTopicScroll()
                self.updateTopicsTabDocScroll()
                # Update Doc Content, Topics & Documents in the Documents Tab.
                cur_doc_id = self.search_engine.random_doc_id()
                self.updateDocsTabVariables(
                    cur_doc_id=cur_doc_id, show_progress=show_progress
                )
                self.updateDocsTabContentArea()
                self.updateDocsTabTopicsScroll()
                self.updateDocsTabDocsScroll()
                # Done.
                if show_progress:
                    progress_msg("Topic Model Updated!")
        # Report that this is the same Model.
        elif show_progress:
            progress_msg("No need to update the Model.")

    def updateSupportedSizes(self):
        """
        Update the Supported Sizes on the App by updating the class attributes
        and the Size ComboBoxes.
        """
        # Save the number of items in the old sizes.
        old_sizes_len = len(self.supported_sizes)
        # Get the Topic Model Sizes Again.
        topic_size = str(self.search_engine.topic_size)
        supported_sizes = [str(x) for x in self.search_engine.supported_model_sizes()]
        cur_size_index = supported_sizes.index(topic_size)
        # Update Class Attributes.
        self.topic_size = topic_size
        self.supported_sizes = supported_sizes
        self.cur_size_index = cur_size_index

        # Signal that we are going to work on the Size ComboBoxes.
        self.size_combo_working = True
        # Remove the Previous Sizes.
        for _ in range(old_sizes_len):
            self.search_tab_size_combo.removeItem(0)
            self.topics_tab_size_combo.removeItem(0)
            self.docs_tab_size_combo.removeItem(0)
        # Add the New Sizes.
        self.search_tab_size_combo.addItems(self.supported_sizes)
        self.search_tab_size_combo.setCurrentIndex(self.cur_size_index)
        self.topics_tab_size_combo.addItems(self.supported_sizes)
        self.topics_tab_size_combo.setCurrentIndex(self.cur_size_index)
        self.docs_tab_size_combo.addItems(self.supported_sizes)
        self.docs_tab_size_combo.setCurrentIndex(self.cur_size_index)
        # We are done working on the Size ComboBoxes.
        self.size_combo_working = False

    def changeTopicSize(self, new_size_index: int, show_progress=False):
        """
        Change the size of the Topic Model to the size in the 'new_size_index'.
        """
        # Check if we are not working on the Size Combos.
        if self.size_combo_working:
            return
        # Signal that we are working on the Size Combos.
        self.size_combo_working = True

        # Check if we have a new Size Index.
        if new_size_index != self.cur_size_index:
            # Create the Progress Dialog.
            progress_dialog = QProgressDialog(
                "Changing Topic Size...", "Cancel", 0, 6, self
            )
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            cancel_button = QPushButton("Cancel")
            progress_dialog.setCancelButton(cancel_button)
            cancel_button.setEnabled(False)
            progress_dialog.setValue(1)

            # Update the Size Combo to the New Size.
            if new_size_index != self.search_tab_size_combo.currentIndex():
                self.search_tab_size_combo.setCurrentIndex(new_size_index)
            if new_size_index != self.topics_tab_size_combo.currentIndex():
                self.topics_tab_size_combo.setCurrentIndex(new_size_index)
            if new_size_index != self.docs_tab_size_combo.currentIndex():
                self.docs_tab_size_combo.setCurrentIndex(new_size_index)
            # Update Progress.
            progress_dialog.setValue(2)

            # Get the New Size.
            new_topic_size = self.supported_sizes[new_size_index]
            # Change the Size of the Topic Model.
            self.search_engine.update_topic_size(
                new_size=new_topic_size, show_progress=show_progress
            )
            # Update Progress.
            progress_dialog.setValue(3)

            # Update Documents & Topics in the Search Tab.
            self.updateSearchTabVariables(show_progress=show_progress)
            self.updateSearchTabTopicsScroll()
            self.updateSearchTabDocsScroll()
            # Update Progress.
            progress_dialog.setValue(4)
            # Update Topics & Documents in the Topics Tab.
            self.updateTopicsTabTopicVariables()
            self.updateTopicsTabDocVariables()
            self.updateTopicsTabTopicScroll()
            self.updateTopicsTabDocScroll()
            # Update Progress.
            progress_dialog.setValue(5)
            # Update Doc Content, Topics & Documents in the Documents Tab.
            self.updateDocsTabVariables(show_progress=show_progress)
            self.updateDocsTabContentArea()
            self.updateDocsTabTopicsScroll()
            self.updateDocsTabDocsScroll()
            # Update Progress.
            progress_dialog.setValue(6)

            # Update the Topic Size Attributes.
            self.topic_size = new_topic_size
            self.cur_size_index = new_size_index
            if show_progress:
                progress_msg(f"Topic Model size updated to {new_topic_size}")

        # Report that this is the same Topic Size.
        elif show_progress:
            progress_msg("No need to update the size of the Topic Model.")
        # We are Done.
        self.size_combo_working = False

    def searchRequested(self):
        """
        The Search Button was touched, if we have a new query term, make a
        search.
        """
        # Check if we have a new Query.
        old_query_text = self.search_tab_query_text
        new_query_text = self.search_tab_line_edit.text()
        if new_query_text != old_query_text:
            # --- We have a New Query ---
            # Signal that we are Working on the Search Tab.
            self.search_tab_working = True
            # Update the Variables of the Search Tab.
            self.updateSearchTabVariables(query_text=new_query_text)
            # Update Content in Widgets & Scrollable.
            self.updateSearchTabDocsScroll()
            self.updateSearchTabTopicsScroll()
            # Done Working on the Search Tab.
            self.search_tab_working = False
        # Update the Placeholder Text with the new Query:
        if self.search_tab_line_edit.placeholderText():
            self.search_tab_line_edit.setPlaceholderText(f" {new_query_text}")
        # Go to the Search Tab if we are not there.
        if self.current_tab_index != self.search_tab_index:
            self.current_tab_index = self.search_tab_index
            self.main_tab_bar.setCurrentIndex(self.current_tab_index)

    def updateSearchTabVariables(self, query_text='', show_progress=False):
        """
        Update the Value of the Variables used in the Search Tab given the text
        of a new query made by the user.
        """
        # Check if we have a new query_text.
        if query_text:
            # Get the Embedding of the Query.
            query_embed = self.search_engine.text_embed(query_text)
        else:
            query_text = self.search_tab_query_text
            query_embed = self.search_tab_query_embed

        # Get the Top Topics and Top Documents (for the top topics).
        topic_num = self.search_tab_topics_num
        doc_num = self.search_tab_docs_num
        top_topics_sims, top_docs_sims = self.search_engine.embed_query(
            embed=query_embed, topic_num=topic_num, doc_num=doc_num,
            vector_space='words', show_progress=show_progress
        )
        # Get Information of the Top Topics.
        topic_word_num = self.search_tab_topics_num
        search_tab_topics_info = [
            (topic_id, topic_sim,
             self.search_engine.topic_description(topic_id, topic_word_num))
            for topic_id, topic_sim in top_topics_sims
        ]
        # Get Information of the Top Documents.
        search_tab_docs_info = [
            (doc_id, doc_sim,
             self.search_engine.doc_title(doc_id),
             self.search_engine.doc_abstract(doc_id))
            for doc_id, doc_sim in top_docs_sims
        ]
        # Save the Top Topic ID.
        search_tab_top_topic, _ = top_topics_sims[0]

        # Update Search Tab - Variables
        self.search_tab_working = False
        self.search_tab_query_text = query_text
        self.search_tab_query_embed = query_embed
        self.search_tab_top_topic = search_tab_top_topic
        self.search_tab_all_topics = False
        self.search_tab_selected_topics = {search_tab_top_topic}
        self.search_tab_topics_info = search_tab_topics_info
        self.search_tab_docs_info = search_tab_docs_info
        self.search_tab_topic_checkboxes = {}

    def updateSearchTabTopicsScroll(self):
        """
        Update the Topics being displayed in the Topics Scrollable in the Search
        Tab.
        """
        # Disable the 'All Topics' Checkbox if it's Enabled.
        self.search_tab_working = True
        if self.search_tab_all_checkbox and self.search_tab_all_checkbox.isChecked():
            self.search_tab_all_checkbox.setChecked(False)
        self.search_tab_working = False

        # Get the Info of the Topics we are going to Show.
        topics_info = self.search_tab_topics_info

        # Create Layout for the List of Topics.
        top_topics_v_box = QVBoxLayout()
        top_topics_v_box.setSpacing(0)
        top_topics_v_box.setContentsMargins(0, 0, 0, 0)
        top_topics_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Add the Topic Items to the Layout.
        for topic_id, similarity, description in topics_info:
            is_checked = topic_id in self.search_tab_selected_topics
            topic_info_container = self.createTopicItem(
                topic_id=topic_id, cat_type='similarity', cat_value=similarity,
                description=description, view_type='topic',
                checkable_type='search-checkbox', set_checked=is_checked
            )
            # Add to Layout.
            top_topics_v_box.addWidget(topic_info_container)
        # Add a Stretch at the end (in case we have only a few items).
        top_topics_v_box.addStretch()
        # Create Container for the List of Items.
        top_topics_container = QWidget()
        top_topics_container.setLayout(top_topics_v_box)

        # Search Tab - Update the Topics Scrollable of Topics.
        self.search_tab_topics_scroll.setWidget(top_topics_container)

    def newSearchTabAllTopics(self, checked: bool, show_progress=False):
        """
        The 'Use All Topics' checkbox has been toggled. Depending on the value
        of 'checked' either show the Top Documents using all the corpus or the
        documents from the top topic for the search query.
        """
        # Do not Respond to the Signal if the Search Tab is Working.
        if self.search_tab_working:
            return
        # Signal that we are working on the Search Tab.
        self.search_tab_working = True
        # Get the Current Query Embedding.
        cur_query_embed = self.search_tab_query_embed

        # Check if the Checkbox has been selected or deselected.
        if checked:
            # --- All Topics Selected ---
            if show_progress:
                progress_msg("All Topics Selected in the Search Tab.")
            self.search_tab_all_topics = True
            # Deselect all the currently selected topics.
            for topic_id in self.search_tab_selected_topics:
                topic_checkbox = self.search_tab_topic_checkboxes[topic_id]
                topic_checkbox.setChecked(False)
            # Clean the Selected Topics Set.
            self.search_tab_selected_topics = None
            # Use the Default List of All the Documents to find the Top Docs.
            doc_ids = None
        else:
            # --- All Topics Checkbox Deselected ---
            if show_progress:
                progress_msg("All Topics Deselected in the Search Tab.")
            self.search_tab_all_topics = False
            # Select the Documents from Top topic.
            top_topic_id = self.search_tab_top_topic
            self.search_tab_selected_topics = {top_topic_id}
            top_topic_checkbox = self.search_tab_topic_checkboxes[top_topic_id]
            top_topic_checkbox.setChecked(True)
            # Use the Documents from Top Topic to find the Top Documents.
            doc_ids = self.search_engine.topic_doc_ids(top_topic_id)

        # Find the Top Documents using the Collected Doc IDs.
        doc_num = self.search_tab_docs_num
        top_docs_sims = self.search_engine.embed_query_top_docs(
            embed=cur_query_embed, doc_ids=doc_ids, doc_num=doc_num,
            vector_space='words', show_progress=show_progress
        )
        # Get the Info about the Top Documents.
        search_tab_docs_info = [
            (doc_id, doc_sim,
             self.search_engine.doc_title(doc_id),
             self.search_engine.doc_abstract(doc_id))
            for doc_id, doc_sim in top_docs_sims
        ]
        # Save the Documents Info.
        self.search_tab_docs_info = search_tab_docs_info
        # Update the Documents Scrollable.
        self.updateSearchTabDocsScroll()
        # Done Working on the Search Tab.
        self.search_tab_working = False

    def newSearchTabTopicSelection(self, checked: bool, topic_id: str):
        """
        Update the Number of Topics selected for the search of documents either
        in the Search Tab or the Documents Tab.
        """
        # Do Not make updates if the Search Tab is Working.
        if self.search_tab_working:
            return
        # Signal that we are working on the Search Tab.
        self.search_tab_working = True
        # Get the Current Query Embedding.
        cur_query_embed = self.search_tab_query_embed

        # Check if this 'topic_id' was selected or deselected.
        if checked:
            # --- New Topic Selected for the Search ---
            if self.search_tab_all_topics:
                # - This is the Only Topic -
                self.search_tab_all_topics = False
                self.search_tab_all_checkbox.setChecked(False)
                self.search_tab_selected_topics = {topic_id}
                # Use the Documents from this Topic.
                doc_ids = self.search_engine.topic_doc_ids(topic_id)
            else:
                # - There are other Selected Topics -
                self.search_tab_selected_topics.add(topic_id)
                # Use the Topics Docs from the Selected Documents.
                doc_ids = []
                for sel_topic_id in self.search_tab_selected_topics:
                    new_doc_ids = self.search_engine.topic_doc_ids(sel_topic_id)
                    doc_ids += new_doc_ids
        else:
            # --- Remove the Current Topic from the Search ---
            if len(self.search_tab_selected_topics) == 1:
                # - This was the only Topic Selected -
                self.search_tab_all_topics = True
                self.search_tab_all_checkbox.setChecked(True)
                self.search_tab_selected_topics = None
                # Use the Default Lists of All Doc IDs in the Topic Model.
                doc_ids = None
            else:
                # - There are still more Topics Selected -
                self.search_tab_selected_topics.remove(topic_id)
                # Use the Topics Docs from the still Selected Documents.
                doc_ids = []
                for sel_topic_id in self.search_tab_selected_topics:
                    new_doc_ids = self.search_engine.topic_doc_ids(sel_topic_id)
                    doc_ids += new_doc_ids

        # Find the Top Documents using the Collected Doc IDs.
        doc_num = self.search_tab_docs_num
        top_docs_sims = self.search_engine.embed_query_top_docs(
            embed=cur_query_embed, doc_ids=doc_ids,
            doc_num=doc_num, vector_space='words'
        )
        # Get the Info about the Top Documents.
        search_tab_docs_info = [
            (doc_id, doc_sim,
             self.search_engine.doc_title(doc_id),
             self.search_engine.doc_abstract(doc_id))
            for doc_id, doc_sim in top_docs_sims
        ]
        # Save the Documents Info.
        self.search_tab_docs_info = search_tab_docs_info
        # Update the Documents Scrollable.
        self.updateSearchTabDocsScroll()
        # Done Working on the Search Tab.
        self.search_tab_working = False

    def updateSearchTabDocsScroll(self):
        """
        Update the Documents being displayed in the Documents Scrollable in the
        Search Tab.
        """
        # Get the Info of the Documents we are going to show.
        docs_info = self.search_tab_docs_info

        # Create the Layout for the List of Documents.
        docs_v_box = QVBoxLayout()
        docs_v_box.setSpacing(0)
        docs_v_box.setContentsMargins(0, 0, 0, 0)
        docs_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Add the Documents Items to the Layout.
        for doc_id, similarity, title, abstract in docs_info:
            text_len = self.search_tab_preview_size
            extra = '...' if text_len < len(abstract) else ''
            red_abstract = abstract[:text_len] + extra
            doc_info_item = self.createDocItem(
                doc_id=doc_id, title=title, abstract=red_abstract, sim=similarity
            )
            docs_v_box.addWidget(doc_info_item)
        # Add a Stretch at the end (in case we have only a few items).
        docs_v_box.addStretch()
        # Create Container for the Item List.
        docs_container = QWidget()
        docs_container.setLayout(docs_v_box)

        # Update the Documents Scrollable in the Search Tab.
        self.search_tab_docs_scroll.setWidget(docs_container)

    def updateTopicsTabTopicVariables(self, cur_cat_index=-1):
        """
        Update the values of the Variables used in the Topics Tab.
        """
        # Check if we have to update the Variables with a new Category.
        if cur_cat_index != -1:
            # Get Index of the Sorting Category.
            cur_sort_cat = self.topics_tab_sort_cats[cur_cat_index]
        else:
            cur_sort_cat = self.topics_tab_cur_cat
            cur_cat_index = self.topics_tab_cat_index

        # Get the Topics Information with the current Sorting Category.
        lower_sort_cat = cur_sort_cat.lower()
        topics_word_num = self.topics_tab_word_num
        cur_cat_topics_info = self.search_engine.topics_info(
            sort_cat=lower_sort_cat, word_num=topics_word_num
        )
        # Get the Top Topic for the Category.
        cur_cat_top_topic, _, _ = cur_cat_topics_info[0]

        # Calculate Number of Topic Pages and Topics per Page.
        topic_page_size = self.topics_tab_topics_page_size
        total_topics = len(cur_cat_topics_info)
        total_topic_pages = total_topics // topic_page_size
        if total_topics % topic_page_size > 0:
            total_topic_pages += 1
        # Save the Topics Info per Page.
        topic_page_count = 0
        current_page = 1
        cur_page_topics_info = []
        topic_pages = {}
        topic_page_topics_info = {}
        for topic_id, cat_value, description in cur_cat_topics_info:
            # Save Page of the Topic.
            topic_pages[topic_id] = str(current_page)
            # Save the Topic Info of the topic.
            topic_info_tuple = (topic_id, cat_value, description)
            cur_page_topics_info.append(topic_info_tuple)
            # Check if we need to go to the next Page.
            topic_page_count += 1
            if topic_page_count >= topic_page_size:
                # Save the Topics Info for the current Page.
                topic_page_topics_info[str(current_page)] = cur_page_topics_info
                # Go to the Next Page & Reset all the Page Variables.
                current_page += 1
                topic_page_count = 0
                cur_page_topics_info = []
        # Check if we have Info <Leftovers>.
        if cur_page_topics_info:
            topic_page_topics_info[str(current_page)] = cur_page_topics_info

        # Update Topics Tab - Topic Variables
        self.topics_tab_cur_cat = cur_sort_cat
        self.topics_tab_cat_index = cur_cat_index
        self.topics_tab_cur_topic = cur_cat_top_topic
        self.topics_tab_topics_cur_page = '1'
        self.topics_tab_topics_total_pages = str(total_topic_pages)
        self.topics_tab_topics_pages = topic_pages
        self.topics_tab_topics_page_info = topic_page_topics_info
        self.topics_tab_button_group = QButtonGroup()
        self.topics_tab_topics_radios = {}
        self.topics_tab_topics_widgets = {}

    def changeTopicSorting(self, sort_cat_index: int, show_progress=False):
        """
        Change the sorting category use to rank the topics in the Topics Tab.
        """
        # Check if we have a new sorting category.
        if sort_cat_index == self.topics_tab_cat_index:
            # Same Sorting Category.
            if show_progress:
                progress_msg(f"Same sorting category selected <{self.topics_tab_cur_cat}>.")
            return
        # Save Old Top Topic to see if it changes.
        old_cur_topic = self.topics_tab_cur_topic

        # --- New Sorting Category Selected ---
        cur_sort_cat = self.topics_tab_sort_cats[sort_cat_index]
        if show_progress:
            progress_msg(
                f"Updating the sorting category of the topics to <{cur_sort_cat}>..."
            )
        # Update Topic Variables & Scrollable
        self.updateTopicsTabTopicVariables(cur_cat_index=sort_cat_index)
        self.updateTopicsTabTopicScroll()
        # Topics Tab - Update Scrollable of Documents if the Top Topic Changed.
        if self.topics_tab_cur_topic != old_cur_topic:
            self.updateTopicsTabDocVariables()
            self.updateTopicsTabDocScroll()
        # Done.
        if show_progress:
            progress_msg(
                f"Topics in the Topics Tab sorted by {self.topics_tab_cur_cat}!"
            )

    def viewTopic(self, topic_id: str):
        """
        Open the Topics Tab to view the Topic 'topic_id'.
        """
        # See if 'topic_id' is in the current Topic Page.
        cur_topic_page = self.topics_tab_topics_cur_page
        new_topic_page = self.topics_tab_topics_pages[topic_id]
        if new_topic_page == cur_topic_page:
            # -- Same Topic Page --
            # Get the Topic RadioButton and Container in Scrollable.
            topic_radio_button = self.topics_tab_topics_radios[topic_id]
            # Select the Given Topic in Scrollable.
            topic_radio_button.setChecked(True)
        else:
            # -- New Topic Page --
            self.topics_tab_cur_topic = topic_id
            self.topics_tab_topics_cur_page = new_topic_page
            self.updateTopicsTabTopicScroll()
            self.updateTopicsTabDocVariables()
            self.updateTopicsTabDocScroll()
        # Make sure to show the Topic Widget.
        topic_widget_item = self.topics_tab_topics_widgets[topic_id]
        self.topics_tab_topics_scroll.ensureWidgetVisible(topic_widget_item)
        # Go to the Topics Tab if we are not there.
        if self.current_tab_index != self.topics_tab_index:
            self.current_tab_index = self.topics_tab_index
            self.main_tab_bar.setCurrentIndex(self.current_tab_index)

    def firstPageTopicsTabTopicScroll(self):
        """
        Move the Topics Scrollable on the Topics Tab to the First Page.
        """
        # Set the current page on the First Page.
        self.topics_tab_topics_cur_page = '1'
        self.updateTopicsTabTopicScroll()

    def nextPageTopicsTabTopicScroll(self):
        """
        Move the Topics Scrollable on the Topics Tab to the Next Page.
        """
        # Increase by 1 the page number.
        cur_page_num = int(self.topics_tab_topics_cur_page)
        cur_page_num += 1
        self.topics_tab_topics_cur_page = str(cur_page_num)
        # Update the Scrollable.
        self.updateTopicsTabTopicScroll()

    def prevPageTopicsTabTopicScroll(self):
        """
        Move the Topics Scrollable on the Topics Tab to the Previous Page.
        """
        # Decrease by 1 the page number.
        cur_page_num = int(self.topics_tab_topics_cur_page)
        cur_page_num -= 1
        self.topics_tab_topics_cur_page = str(cur_page_num)
        # Update the Scrollable.
        self.updateTopicsTabTopicScroll()

    def lastPageTopicsTabTopicsScroll(self):
        """
        Move the Topics Scrollable on the Topics Tab to the Last Page.
        """
        # Set the current page on the Last Page.
        last_page = self.topics_tab_topics_total_pages
        self.topics_tab_topics_cur_page = last_page
        self.updateTopicsTabTopicScroll()

    def updateTopicsTabTopicScroll(self):
        """
        Create or Update the Content of the Topics being displayed in the
        Topics Scrollable in the Topics Tab.
        """
        # Reset the Button Group & Dictionaries with the Button & Widgets.
        self.topics_tab_button_group = QButtonGroup()
        self.topics_tab_topics_radios = {}
        self.topics_tab_topics_widgets = {}

        # Create Layout for the List of Topics.
        top_topics_v_box = QVBoxLayout()
        top_topics_v_box.setSpacing(0)
        top_topics_v_box.setContentsMargins(0, 0, 0, 0)
        top_topics_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Get the Sorting Category and the Topics Info.
        lower_sort_cat = self.topics_tab_cur_cat.lower()
        cur_topic_id = self.topics_tab_cur_topic
        cur_topic_page = self.topics_tab_topics_cur_page
        cur_page_topics_info = self.topics_tab_topics_page_info[cur_topic_page]
        # Create the Widget Items for the Topics in the Current Page.
        for topic_id, cat_value, description in cur_page_topics_info:
            is_cur_topic = topic_id == cur_topic_id
            topic_info_item = self.createTopicItem(
                topic_id=topic_id, cat_type=lower_sort_cat, cat_value=cat_value,
                description=description, view_type='vocabulary',
                checkable_type='radio-button', set_checked=is_cur_topic
            )
            # Save the Widget Item of the Topics in the Page.
            self.topics_tab_topics_widgets[topic_id] = topic_info_item
            # Add to Layout.
            top_topics_v_box.addWidget(topic_info_item)
        # Add a Stretch at the end (in case we have only a few items).
        top_topics_v_box.addStretch()
        # Create Container for the List of Topics.
        top_topics_container = QWidget()
        top_topics_container.setLayout(top_topics_v_box)

        # Topics Tab - Update Scrollable of Topics.
        self.topics_tab_topics_scroll.setWidget(top_topics_container)
        # Update the Topics Page Labels.
        cur_page = int(self.topics_tab_topics_cur_page)
        total_pages = int(self.topics_tab_topics_total_pages)
        self.topics_tab_topics_page_label.setText(f"Page: {cur_page}/{total_pages}")
        # Update Page Buttons.
        if cur_page > 1:
            self.topics_tab_topics_first_button.setEnabled(True)
            self.topics_tab_topics_prev_button.setEnabled(True)
        else:
            self.topics_tab_topics_first_button.setEnabled(False)
            self.topics_tab_topics_prev_button.setEnabled(False)
        if cur_page < total_pages:
            self.topics_tab_topics_next_button.setEnabled(True)
            self.topics_tab_topics_last_button.setEnabled(True)
        else:
            self.topics_tab_topics_next_button.setEnabled(False)
            self.topics_tab_topics_last_button.setEnabled(False)

    def newTopicSelected(self, checked: bool, topic_id: str, show_progress=False):
        """
        Update the Information being displayed in the Topics Tab, using the new
        Topic selected.
        """
        # Check if this the new enabled Topic.
        if not checked:
            # No need to do any changes to the UI.
            return
        # Update the Current Topic in the Topics Tab.
        self.topics_tab_cur_topic = topic_id
        # Update the Displayed Documents.
        self.updateTopicsTabDocVariables()
        self.updateTopicsTabDocScroll()
        # Done.
        if show_progress:
            progress_msg(f"Documents from <{topic_id}> ready!")

    def updateTopicsTabDocVariables(self):
        """
        Update the value of the variables used in the Documents Section of the
        Topics Tab.
        """
        # Get the Doc Info for the documents in the Current Topic.
        cur_topic_docs_info = self.search_engine.topic_docs_info(
            topic_id=self.topics_tab_cur_topic
        )
        # Calculate Number of Document Pages and Docs per Page.
        doc_page_size = self.topics_tab_docs_page_size
        total_docs = len(cur_topic_docs_info)
        total_doc_pages = total_docs // doc_page_size
        if total_docs % doc_page_size > 0:
            total_doc_pages += 1
        # Save the Documents Info per Page.
        doc_page_count = 0
        current_page = 1
        current_page_docs_info = []
        docs_page_info = {}
        for doc_info_tuple in cur_topic_docs_info:
            # Add The Info Tuple to the Current Doc Info List.
            current_page_docs_info.append(doc_info_tuple)
            # Check if we need to go to the next Page.
            doc_page_count += 1
            if doc_page_count >= doc_page_size:
                # Save the Docs Info for the Current Page.
                docs_page_info[str(current_page)] = current_page_docs_info
                # Go to Next Page & Reset all the Page Variables.
                current_page += 1
                doc_page_count = 0
                current_page_docs_info = []
        # Check for any Doc Info <Leftovers>.
        if current_page_docs_info:
            docs_page_info[str(current_page)] = current_page_docs_info

        # Update Topics Tab - Doc Variables.
        self.topics_tab_docs_num = len(cur_topic_docs_info)
        self.topics_tab_docs_cur_page = '1'
        self.topics_tab_docs_total_pages = str(total_doc_pages)
        self.topics_tab_docs_page_info = docs_page_info

    def firstPageTopicsTabDocScroll(self):
        """
        Move the Documents Scrollable on the Topics Tab to the First Page.
        """
        # Set the current page on the First Page.
        self.topics_tab_docs_cur_page = '1'
        self.updateTopicsTabDocScroll()

    def nextPageTopicsTabDocScroll(self):
        """
        Move the Documents Scrollable on the Topics Tab to the Next Page.
        """
        # Increase by 1 the page number.
        cur_page_num = int(self.topics_tab_docs_cur_page)
        cur_page_num += 1
        self.topics_tab_docs_cur_page = str(cur_page_num)
        # Update the Documents Scrollable.
        self.updateTopicsTabDocScroll()

    def prevPageTopicsTabDocScroll(self):
        """
        Move the Documents Scrollable on the Topics Tab to the Previous Page.
        """
        # Decrease by 1 the page number.
        cur_page_num = int(self.topics_tab_docs_cur_page)
        cur_page_num -= 1
        self.topics_tab_docs_cur_page = str(cur_page_num)
        # Update the Documents Scrollable.
        self.updateTopicsTabDocScroll()

    def lastPageTopicsTabDocScroll(self):
        """
        Move the Documents Scrollable on the Topics Tab to the Last Page.
        """
        # Set the current page on the Last Page.
        last_page = self.topics_tab_docs_total_pages
        self.topics_tab_docs_cur_page = last_page
        self.updateTopicsTabDocScroll()

    def updateTopicsTabDocScroll(self):
        """
        Create & Update the Content of the Documents Scrollable in the Topics
        Tab.
        """
        # Get the Widget Items of the Documents in the Current Page.
        doc_page_key = self.topics_tab_docs_cur_page
        cur_page_docs_info = self.topics_tab_docs_page_info[doc_page_key]

        # Topic Documents Layout.
        top_docs_v_box = QVBoxLayout()
        top_docs_v_box.setSpacing(0)
        top_docs_v_box.setContentsMargins(0, 0, 0, 0)
        top_docs_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Add items to Top Documents Layout.
        for doc_id, similarity, title, abstract in cur_page_docs_info:
            # Create the Item Widget for the Document.
            text_len = self.topics_tab_preview_size
            extra = '...' if text_len < len(abstract) else ''
            red_abstract = abstract[:text_len] + extra
            doc_info_item = self.createDocItem(
                doc_id=doc_id, title=title, abstract=red_abstract, sim=similarity
            )
            top_docs_v_box.addWidget(doc_info_item)
        # Add a Stretch at the end (in case we have only a few items).
        top_docs_v_box.addStretch()
        # Create Container for the Documents Layout.
        top_docs_container = QWidget()
        top_docs_container.setLayout(top_docs_v_box)

        # Update the Label the Documents.
        cur_topic_id = self.topics_tab_cur_topic
        cur_topic_size = self.topics_tab_docs_num
        self.topics_tab_docs_label.setText(
            f"Documents from {cur_topic_id} ({cur_topic_size} docs):"
        )
        # Update Documents Scrollable.
        self.topics_tab_docs_scroll.setWidget(top_docs_container)
        # Update the Page Labels.
        cur_page = int(self.topics_tab_docs_cur_page)
        total_pages = int(self.topics_tab_docs_total_pages)
        self.topics_tab_docs_page_label.setText(f"Page: {cur_page}/{total_pages}")
        # Update Page Buttons.
        if cur_page > 1:
            self.topics_tab_docs_first_button.setEnabled(True)
            self.topics_tab_docs_prev_button.setEnabled(True)
        else:
            self.topics_tab_docs_first_button.setEnabled(False)
            self.topics_tab_docs_prev_button.setEnabled(False)
        if cur_page < total_pages:
            self.topics_tab_docs_next_button.setEnabled(True)
            self.topics_tab_docs_last_button.setEnabled(True)
        else:
            self.topics_tab_docs_next_button.setEnabled(False)
            self.topics_tab_docs_last_button.setEnabled(False)

    def viewVocabulary(self, topic_id: str):
        """
        Show a new Dialog with the Vocabulary of the given Topic 'topic_id'.
        """
        progress_msg("Opening Vocabulary...")
        word_list = self.search_engine.topic_vocab(topic_id)
        self.vocab_window = QVocabDialog(topic_id, word_list)
        self.vocab_window.show()

    def getDocID(self):
        """
        Open a Dialog to get the ID to open a new Document.
        """
        progress_msg("Opening Dialog to get Doc ID...")
        doc_ids = self.search_engine.system_doc_ids
        id_window = QOpenDocument(all_doc_ids=doc_ids)
        if id_window.exec() == QDialog.DialogCode.Accepted:
            doc_id = id_window.doc_id
            self.viewDocument(doc_id=doc_id)

    def openRandomDoc(self):
        """
        Open a Random Document.
        """
        random_doc_id = self.search_engine.random_doc_id()
        self.viewDocument(doc_id=random_doc_id)

    def viewDocument(self, doc_id: str):
        """
        Show a Document in the Document Tab.
        """
        # Check if it is the same document.
        if doc_id != self.docs_tab_cur_doc:
            # --- Open the New Document --
            # Signal that we are Working the Documents Tab.
            self.docs_tab_working = True
            # Update the Variables of the Documents Tab.
            self.updateDocsTabVariables(cur_doc_id=doc_id)
            # Update Content in Widgets & Scrollable.
            self.updateDocsTabTopicsScroll()
            self.updateDocsTabContentArea()
            self.updateDocsTabDocsScroll()
            # Done Working on the Documents Tab.
            self.docs_tab_working = False

        # Go to the Documents Tab if we are not there.
        if self.current_tab_index != self.docs_tab_index:
            self.current_tab_index = self.docs_tab_index
            self.main_tab_bar.setCurrentIndex(self.current_tab_index)

    def updateDocsTabVariables(self, cur_doc_id='', show_progress=False):
        """
        Update the Values of the Variables used to display information on the
        Documents Tab based on the New Document 'doc_id'.
        """
        # Check if we have to update the Current Document.
        if cur_doc_id:
            # Get the Embedding & Full Content of the Document.
            cur_doc_embed = self.search_engine.doc_embed(cur_doc_id, space='documents')
            doc_full_content = self.search_engine.doc_full_content(cur_doc_id)
        else:
            # Using the Same Document.
            cur_doc_id = self.docs_tab_cur_doc
            cur_doc_embed = self.docs_tab_doc_embed
            doc_full_content = self.docs_tab_doc_content

        # Get the Top Topics and Close Documents Information.
        # Doc Num (+1) in case the search returns the document itself.
        top_topics_sims, top_docs_sims = self.search_engine.embed_query(
            embed=cur_doc_embed, topic_num=self.docs_tab_topics_num,
            doc_num=(self.docs_tab_docs_num + 1), vector_space='documents',
            show_progress=show_progress
        )
        docs_tab_topics_info = [
            (topic_id, topic_sim,
             self.search_engine.topic_description(
                 topic_id=topic_id, word_num=self.docs_tab_word_num
             ))
            for topic_id, topic_sim in top_topics_sims
        ]
        docs_tab_docs_info = [
            (doc_id, doc_sim,
             self.search_engine.doc_title(doc_id),
             self.search_engine.doc_abstract(doc_id))
            for doc_id, doc_sim in top_docs_sims
            if doc_id != cur_doc_id
        ]
        # Make sure we have the right amount of docs.
        docs_tab_docs_info = docs_tab_docs_info[:self.docs_tab_docs_num]
        # Get the Top Topic.
        docs_tab_top_topic, _ = top_topics_sims[0]

        # Update Documents Tab - Variables
        self.docs_tab_working = False
        self.docs_tab_cur_doc = cur_doc_id
        self.docs_tab_doc_embed = cur_doc_embed
        self.docs_tab_doc_content = doc_full_content
        self.docs_tab_top_topic = docs_tab_top_topic
        self.docs_tab_all_topics = False
        self.docs_tab_selected_topics = {docs_tab_top_topic}
        self.docs_tab_topics_info = docs_tab_topics_info
        self.docs_tab_docs_info = docs_tab_docs_info
        self.docs_tab_topic_checkboxes = {}

    def updateDocsTabContentArea(self):
        """
        Create or Update the Content of the Text Widget with the current
        Document in the Documents Tab.
        """
        # Get Current Doc ID.
        doc_id = self.docs_tab_cur_doc
        # Create Text for the Name Label.
        header_text = f"""
            <p align='left'><font face='Times New Roman' size='+1'>
            Content of Document &lt;{doc_id}&gt;
            </font></p>
        """
        # Get the Title and Abstract.
        title = self.search_engine.doc_title(doc_id)
        abstract = self.search_engine.doc_abstract(doc_id)
        # Create Content.
        content = f"""
            <p align='justify'><font face='Times New Roman' size='+2'><u>
            {title}
            </u></font></p>
            <p align='justify'><font size='+1'>{abstract}</font></p>
        """

        # Update the Full Content Button.
        has_full_content = self.docs_tab_doc_content != ''
        self.docs_tab_content_button.setEnabled(has_full_content)
        # Set the Text for the Name Label.
        self.docs_tab_name_label.setText(header_text)
        # Set the Content inside the TextEdit Widget.
        self.docs_tab_text_widget.setText(content)

    def updateDocsTabTopicsScroll(self):
        """
        Create or Update the Content of the Topics being displayed in the Topics
        Scrollable in the Documents Tab.
        """
        # Disable the 'All Topics' Checkbox if it's Enabled.
        self.docs_tab_working = True
        if self.docs_tab_all_checkbox and self.docs_tab_all_checkbox.isChecked():
            self.docs_tab_all_checkbox.setChecked(False)
        self.docs_tab_working = False

        # Create Layout for the List of Topics.
        top_topics_v_box = QVBoxLayout()
        top_topics_v_box.setSpacing(0)
        top_topics_v_box.setContentsMargins(0, 0, 0, 0)
        top_topics_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Create the Topics Info Items & Add them to the Layout.
        topics_info = self.docs_tab_topics_info
        # Mark the first Topic of the List.
        for topic_id, similarity, description in topics_info:
            is_checked = topic_id in self.docs_tab_selected_topics
            topic_info_container = self.createTopicItem(
                topic_id=topic_id, cat_type='similarity', cat_value=similarity,
                description=description, view_type='topic',
                checkable_type='doc-checkbox', set_checked=is_checked
            )
            # Add to Layout.
            top_topics_v_box.addWidget(topic_info_container)
        # Add a Stretch at the end (in case we have only a few items).
        top_topics_v_box.addStretch()
        # Create Container for the List of Items.
        top_topics_container = QWidget()
        top_topics_container.setLayout(top_topics_v_box)

        # Doc Tab - Update Scrollable of Topics.
        self.docs_tab_topics_scroll.setWidget(top_topics_container)

    def newDocsTabAllTopics(self, checked: bool, show_progress=False):
        """
        The 'Use All Topics' checkbox has been toggled. Depending on the value
        of 'checked' either show the Top Documents using all the corpus or the
        documents from the top topic for the document.
        """
        # Do not Respond to the Signal if the Documents Tab is Busy.
        if self.docs_tab_working:
            return
        # Signal that we are Working the Documents Tab.
        self.docs_tab_working = True

        # Get the Current Doc Info.
        cur_doc_id = self.docs_tab_cur_doc
        cur_doc_embed = self.docs_tab_doc_embed

        # Check if the Checkbox was checked or deselected.
        if checked:
            # --- All Topics Selected ---
            if show_progress:
                progress_msg("All Topics Selected in the Documents Tab.")
            self.docs_tab_all_topics = True
            # Deselected all the currently selected topics.
            for topic_id in self.docs_tab_selected_topics:
                topic_checkbox = self.docs_tab_topic_checkboxes[topic_id]
                topic_checkbox.setChecked(False)
            # Clean the Selected Topics Set.
            self.docs_tab_selected_topics = None
            # Use the Default List of All the Documents to find the Top Docs.
            doc_ids = None
        else:
            # --- All Topics Deselected ---
            if show_progress:
                progress_msg("All Topics De-selected in the Documents Tab.")
            self.docs_tab_all_topics = False
            # Select the Document's Topic (Top Topic).
            top_topic_id = self.docs_tab_top_topic
            self.docs_tab_selected_topics = {top_topic_id}
            top_topic_checkbox = self.docs_tab_topic_checkboxes[top_topic_id]
            top_topic_checkbox.setChecked(True)
            # Use the Documents of the Top Topic to find the Top Documents.
            doc_ids = self.search_engine.topic_doc_ids(top_topic_id)

        # Find the Top Docs using the documents in the Top Topic.
        # Doc Num (+1) in case the search returns the document itself.
        top_docs_sims = self.search_engine.embed_query_top_docs(
            embed=cur_doc_embed, doc_ids=doc_ids,
            doc_num=(self.docs_tab_docs_num + 1),
            vector_space='documents', show_progress=show_progress
        )
        docs_tab_docs_info = [
            (doc_id, doc_sim,
             self.search_engine.doc_title(doc_id),
             self.search_engine.doc_abstract(doc_id))
            for doc_id, doc_sim in top_docs_sims
            if doc_id != cur_doc_id
        ]
        # Make sure we have the right amount of docs.
        docs_tab_docs_info = docs_tab_docs_info[:self.docs_tab_docs_num]
        # Save the Documents Info.
        self.docs_tab_docs_info = docs_tab_docs_info
        # Update the Documents Scrollable.
        self.updateDocsTabDocsScroll()
        # Done Working on the Documents Tab.
        self.docs_tab_working = False

    def newDocsTabTopicSelection(
            self, checked: bool, topic_id: str, show_progress=False
    ):
        """
        Update the Topics Selected for the list of Close Documents in regard to
        the current document we are viewing.
        """
        # Do not Respond to the Signal if the Documents Tab is Busy.
        if self.docs_tab_working:
            return
        # Signal that we are Working the Documents Tab.
        self.docs_tab_working = True

        # Get the Current Doc Info.
        cur_doc_id = self.docs_tab_cur_doc
        cur_doc_embed = self.docs_tab_doc_embed

        # Check if the Checkbox was checked or deselected.
        if checked:
            # --- New Topic Selected for the Search ---
            if show_progress:
                progress_msg(f"New Topic Selected: {topic_id}.")
            if self.docs_tab_all_topics:
                # - This is the Only Topic -
                self.docs_tab_all_topics = False
                self.docs_tab_all_checkbox.setChecked(False)
                self.docs_tab_selected_topics = {topic_id}
                # Use the Topic Docs to find the Top Documents.
                doc_ids = self.search_engine.topic_doc_ids(topic_id)
            else:
                # - They Are Other Selected Topics -
                self.docs_tab_selected_topics.add(topic_id)
                # Use the Topic Docs from the Selected Documents.
                doc_ids = []
                for sel_topic_id in self.docs_tab_selected_topics:
                    new_doc_ids = self.search_engine.topic_doc_ids(sel_topic_id)
                    doc_ids += new_doc_ids
        else:
            # --- Remove the Topic from the Search ---
            if show_progress:
                progress_msg(f"Topic Removed: {topic_id}")
            if len(self.docs_tab_selected_topics) == 1:
                # - This was the Only Topic (Now Use All Topics) -
                self.docs_tab_all_topics = True
                self.docs_tab_all_checkbox.setChecked(True)
                self.docs_tab_selected_topics = None
                # Use te Default List of Docs IDs (All from the Topic Model)
                doc_ids = None
            else:
                # - They are still more Topics Selected -
                self.docs_tab_selected_topics.remove(topic_id)
                # Use the Topic Docs from the Selected Documents.
                doc_ids = []
                for sel_topic_id in self.docs_tab_selected_topics:
                    new_doc_ids = self.search_engine.topic_doc_ids(sel_topic_id)
                    doc_ids += new_doc_ids

        # Find the Top Docs using the documents in the Top Topic.
        # Doc Num (+1) in case the search returns the document itself.
        top_docs_sims = self.search_engine.embed_query_top_docs(
            embed=cur_doc_embed, doc_ids=doc_ids,
            doc_num=(self.docs_tab_docs_num + 1),
            vector_space='documents', show_progress=show_progress
        )
        docs_tab_docs_info = [
            (doc_id, doc_sim,
             self.search_engine.doc_title(doc_id),
             self.search_engine.doc_abstract(doc_id))
            for doc_id, doc_sim in top_docs_sims
            if doc_id != cur_doc_id
        ]
        # Make sure we have the right amount of docs.
        docs_tab_docs_info = docs_tab_docs_info[:self.docs_tab_docs_num]
        # Save the Documents Info.
        self.docs_tab_docs_info = docs_tab_docs_info
        # Update the Documents Scrollable.
        self.updateDocsTabDocsScroll()
        # Done Working on the Documents Tab.
        self.docs_tab_working = False

    def updateDocsTabDocsScroll(self):
        """
        Create or Update the Content for the Documents Scrollable in the
        Documents Tab.
        """
        # Get the Info we are going to show.
        docs_info = self.docs_tab_docs_info

        # Create the Documents Layout.
        docs_v_box = QVBoxLayout()
        docs_v_box.setSpacing(0)
        docs_v_box.setContentsMargins(0, 0, 0, 0)
        docs_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Add Doc items to the Layout.
        for doc_id, similarity, title, abstract in docs_info:
            text_len = self.docs_tab_preview_size
            extra = '...' if text_len < len(abstract) else ''
            red_abstract = abstract[:text_len] + extra
            doc_info_item = self.createDocItem(
                doc_id=doc_id, title=title, abstract=red_abstract, sim=similarity
            )
            docs_v_box.addWidget(doc_info_item)
        # Add a Stretch at the end (in case we have only a few items).
        docs_v_box.addStretch()
        # Create Container for the Item List.
        docs_container = QWidget()
        docs_container.setLayout(docs_v_box)

        # Update the Document Scrollable inside the Documents Tab.
        self.docs_tab_docs_scroll.setWidget(docs_container)

    def viewFullContent(self, doc_id: str):
        """
        Open a new Dialog to see the content of the Document 'doc_id'.
        """
        progress_msg("Opening Full Content...")
        full_content = self.docs_tab_doc_content
        self.content_window = QFullContent(doc_id, full_content)
        self.content_window.show()


# Run Application.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(show_progress=True)
    window.show()
    # # Show Window Size after Showing.
    # print(f"Window Width: {window.width()}")
    # print(f"Window Height: {window.height()}")
    sys.exit(app.exec())
