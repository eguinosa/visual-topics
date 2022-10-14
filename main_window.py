# Gelin Eguinosa Rosique
# 2022

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QTabWidget, QLayout, QScrollArea, QLineEdit, QPushButton, QFrame,
    QButtonGroup, QCheckBox, QRadioButton, QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from ir_system import IRSystem
from qvocab_dialog import QVocabDialog
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

        # Gather Data Related with the Current Topic Model.
        topic_size = str(search_engine.topic_size)
        supported_sizes = [str(x) for x in search_engine.supported_model_sizes()]
        cur_size_index = supported_sizes.index(topic_size)
        # Topics Sorting Categories.
        sort_categories = ['Size', 'PWI-tf-idf', 'PWI-exact']
        cur_sort_cat = 'size'
        cur_cat_index = 0
        # Number of Words to describe topics in the Topics Tab.
        topics_tab_word_num = 15
        # Sorting Category Topics Information.
        cur_cat_topics_info = search_engine.topics_info(
            sort_cat=cur_sort_cat, word_num=topics_tab_word_num
        )
        # Current Topic in the Topics Tab.
        top_topic, _, _ = cur_cat_topics_info[0]
        topics_tab_cur_topic = top_topic

        # Document Tab - Number of Top Topics and Documents.
        docs_tab_topics_num = 10
        docs_tab_docs_num = 10
        docs_tab_word_num = 10
        # Gather Data from Random Doc for the Documents Tab.
        # Doc Num (+1) in case the search returns the document itself.
        cur_doc_id, cur_doc_embed = search_engine.random_doc_and_embed()
        cur_doc_full_content = search_engine.doc_full_content(cur_doc_id)
        top_topics_sims, top_docs_sims = search_engine.embed_query(
            embed=cur_doc_embed, topic_num=docs_tab_topics_num,
            doc_num=(docs_tab_docs_num + 1), vector_space='documents',
            show_progress=False
        )
        # Remove the Doc Himself from the List.
        docs_tab_topics_info = [
            (topic_id, topic_sim,
             search_engine.topic_description(topic_id, word_num=docs_tab_word_num))
            for topic_id, topic_sim in top_topics_sims
        ]
        docs_tab_docs_info = [
            (doc_id, doc_sim,
             search_engine.doc_title(doc_id), search_engine.doc_abstract(doc_id))
            for doc_id, doc_sim in top_docs_sims
            if doc_id != cur_doc_id
        ]
        # Make sure we have the right amount of docs.
        docs_tab_docs_info = docs_tab_docs_info[:docs_tab_docs_num]
        # Topics Selected for the Documents Scrollable.
        docs_tab_top_topic, _ = top_topics_sims[0]
        docs_tab_all_topics = False
        docs_tab_selected_topics = {docs_tab_top_topic}

        # Save Class Attributes.
        self.current_model = current_model
        self.search_engine = search_engine
        # Topic Attributes.
        self.topic_size = topic_size
        self.supported_sizes = supported_sizes
        self.cur_size_index = cur_size_index
        # Sorting Categories.
        self.sort_categories = sort_categories
        self.cur_sort_cat = cur_sort_cat
        self.cur_cat_index = cur_cat_index
        self.cur_cat_topics_info = cur_cat_topics_info

        # Search Tab - Information
        self.search_tab_index = 0
        self.search_tab_size_combo = None
        self.search_tab_topics_scroll = None
        # Topics Tab - Information
        self.topics_tab_index = 1
        self.topics_tab_cur_topic = topics_tab_cur_topic
        self.topics_tab_word_num = topics_tab_word_num
        self.topics_tab_button_group = QButtonGroup()
        self.topics_tab_size_combo = None
        self.topics_tab_topics_scroll = None
        self.topics_tab_docs_label = None
        self.topics_tab_docs_scroll = None
        # Documents Tab - Information
        self.docs_tab_index = 2
        self.docs_tab_working = False
        self.docs_tab_topics_num = docs_tab_topics_num
        self.docs_tab_docs_num = docs_tab_docs_num
        self.docs_tab_word_num = docs_tab_word_num
        self.docs_tab_cur_doc = cur_doc_id
        self.docs_tab_doc_embed = cur_doc_embed
        self.docs_tab_doc_content = cur_doc_full_content
        self.docs_tab_all_topics = docs_tab_all_topics
        self.docs_tab_selected_topics = docs_tab_selected_topics
        self.docs_tab_top_topic = docs_tab_top_topic
        self.docs_tab_all_checkbox = None
        self.docs_tab_topic_checkboxes = {}
        self.docs_tab_topics_info = docs_tab_topics_info
        self.docs_tab_docs_info = docs_tab_docs_info
        self.docs_tab_size_combo = None
        self.docs_tab_topics_scroll = None
        self.docs_tab_content_label = None
        self.docs_tab_text_widget = None
        self.docs_tab_docs_scroll = None
        # Main Window - Information
        self.default_tab_index = self.docs_tab_index

        # Menu Bar Actions.
        self.quit_act = None

        # Dialog Windows.
        self.vocab_window = None

        # Create App UI.
        self.initializeUI(show_progress=show_progress)

    def initializeUI(self, show_progress=False):
        """
        Set up the Application's GUI
        """
        # Create Window Size and Name.
        self.setMinimumSize(1100, 800)
        # self.setGeometry(50, 10, 1100, 800)
        self.setWindowTitle("Scientific Search")
        # Change the Location of the Window.
        self.move(10, 80)
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
        tab_bar.addTab(search_tab, "Search")
        tab_bar.addTab(topics_tab, "Topics")
        tab_bar.addTab(docs_tab, "Documents")
        tab_bar.setCurrentIndex(self.default_tab_index)

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
        self.quit_act.setShortcut("Ctr+Q")
        # noinspection PyTypeChecker
        self.quit_act.triggered.connect(self.close)
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
        if show_progress:
            progress_msg("Menu Created!")

    def createSearchTab(self, show_progress=False):
        """
        Create the Widget with the layout in the Search Tab.
        """
        # --- Topic Area ---
        # Get Topic Information
        topic_sizes = [str(x) for x in self.search_engine.supported_model_sizes()]
        topics_info = self.search_engine.topics_info()
        current_size = str(self.search_engine.topic_size)
        index_size = topic_sizes.index(current_size)
        # Topic Size Area.
        size_label = QLabel("Topic Size:")
        size_combo = QComboBox()
        size_combo.addItems(topic_sizes)
        size_combo.setCurrentIndex(index_size)
        size_combo.activated.connect(
            lambda index: self.changeTopicSize(index, show_progress=show_progress)
        )
        # Topic Size Container.
        size_h_box = QHBoxLayout()
        size_h_box.addWidget(size_label)
        size_h_box.addWidget(size_combo)
        size_h_box.addStretch()
        size_container = QWidget()
        size_container.setLayout(size_h_box)
        # Top Topics Area.
        top_topics_label = QLabel("Top Topics:")
        top_topics_v_box = QVBoxLayout()
        top_topics_v_box.setSpacing(0)
        top_topics_v_box.setContentsMargins(0, 0, 0, 0)
        top_topics_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Add items to Topics Layout.
        for topic_id, size, description in topics_info:
            topic_info_container = self.topicInfoItem(
                topic_id=topic_id, cat_type='size',
                cat_value=size, description=description
            )
            top_topics_v_box.addWidget(topic_info_container)
        # Create Scrollable Area with the topics.
        top_topics_container = QWidget()
        top_topics_container.setLayout(top_topics_v_box)
        topics_scroll_area = QScrollArea()
        topics_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        topics_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        topics_scroll_area.setWidget(top_topics_container)
        topics_scroll_area.setWidgetResizable(True)
        # -- Topics Final Layout --
        topics_layout = QVBoxLayout()
        topics_layout.addWidget(size_container)
        topics_layout.addWidget(top_topics_label)
        topics_layout.addWidget(topics_scroll_area)
        topics_container = QWidget()
        topics_container.setLayout(topics_layout)

        # --- Search & Documents Area ---
        # Get Documents Information from the biggest topic in the corpus.
        first_topic_id, _, _ = topics_info[0]
        doc_num = 20
        docs_info = self.search_engine.topic_docs_info(topic_id=first_topic_id)
        # Docs Info - Crop the number of documents we are displaying.
        docs_info = docs_info[:doc_num]
        # Search Layout.
        search_edit = QLineEdit()
        search_button = QPushButton("Search")
        search_layout = QHBoxLayout()
        search_layout.addWidget(search_edit)
        search_layout.addWidget(search_button)
        search_container = QWidget()
        search_container.setLayout(search_layout)
        # Top Documents Layout.
        top_docs_label = QLabel("Top Documents:")
        top_docs_v_box = QVBoxLayout()
        top_docs_v_box.setSpacing(0)
        top_docs_v_box.setContentsMargins(0, 0, 0, 0)
        top_docs_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Add items to Top Documents layout.
        for doc_id, similarity, title, abstract in docs_info:
            doc_info_container = self.docInfoItem(
                doc_id=doc_id, sim=similarity, title=title, abstract=abstract
            )
            top_docs_v_box.addWidget(doc_info_container)
        # Create Scrollable Area with the Documents.
        top_docs_container = QWidget()
        top_docs_container.setLayout(top_docs_v_box)
        docs_scroll_area = QScrollArea()
        docs_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        docs_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        docs_scroll_area.setWidget(top_docs_container)
        docs_scroll_area.setWidgetResizable(True)
        # -- Search & Docs Final Layout --
        search_docs_layout = QVBoxLayout()
        search_docs_layout.addWidget(search_container)
        search_docs_layout.addWidget(top_docs_label)
        search_docs_layout.addWidget(docs_scroll_area)
        search_docs_container = QWidget()
        search_docs_container.setLayout(search_docs_layout)

        # --- Search Tab Layout ---
        search_tab_layout = QHBoxLayout()
        search_tab_layout.addWidget(search_docs_container, 2)
        search_tab_layout.addWidget(topics_container, 1)
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
        sort_cats_combo.addItems(self.sort_categories)
        sort_cats_combo.setCurrentIndex(self.cur_cat_index)
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
        # Update content of Scrollable.
        self.updateTopicsTabTopicScroll()
        # -- All the Topic Area Layout --
        topic_area_layout = QVBoxLayout()
        topic_area_layout.addLayout(size_layout)
        topic_area_layout.addLayout(sort_cats_layout)
        topic_area_layout.addWidget(topics_scroll_area)
        topic_area_container = QWidget()
        topic_area_container.setLayout(topic_area_layout)

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
        # Update content of Scrollable.
        self.updateTopicsTabDocScroll()
        # - Top Docs Area Final Layout -
        docs_area_layout = QVBoxLayout()
        docs_area_layout.addWidget(self.topics_tab_docs_label)
        docs_area_layout.addWidget(self.topics_tab_docs_scroll)
        docs_area_container = QWidget()
        docs_area_container.setLayout(docs_area_layout)

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
        header_text = f"""
            <p align='left'><font face='Times New Roman' size='+1'>
            Content of Document &lt;{str(doc_id)}&gt;
            </font></p>
        """
        # Header Area.
        name_label = QLabel(header_text)
        name_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        full_content_button = QPushButton("Full Content")
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
        self.updateDocTextWidget()
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

    def topicInfoItem(
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
            if set_checked:
                header_checkable.setChecked(True)
            self.topics_tab_button_group.addButton(header_checkable)
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

    def docInfoItem(self, doc_id: str, title: str, abstract='', sim=0):
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

    def changeTopicSize(self, size_index: int, show_progress=False):
        """
        Change the size of the Topic Model to the size in the 'new_size_index'.
        """
        progress_msg("Change Size NOT IMPLEMENTED!!!")

    def newSearchTabTopicSelection(self, checked: bool, topic_id: str):
        """
        Update the Number of Topics selected for the search of documents either
        in the Search Tab or the Documents Tab.
        """
        if checked:
            progress_msg(f"The {topic_id} was selected.")
        elif not checked:
            progress_msg(f"The {topic_id} was deselected.")
        else:
            progress_msg(f"The {topic_id} is partially checked or another not "
                         f"supported action.")

    def viewTopic(self, topic_id: str):
        """
        Open the Topics Tab to view the Topic 'topic_id'.
        """
        progress_msg("View Topic is NOT YET IMPLEMENTED!!!")

    def changeTopicSorting(self, sort_cat_index: int, show_progress=False):
        """
        Change the sorting category use to rank the topics in the Topics Tab.
        """
        # Check if we have a new sorting category.
        if sort_cat_index == self.cur_cat_index:
            # Same Sorting Category.
            if show_progress:
                progress_msg(f"Same sorting category selected <{self.cur_sort_cat}>.")
            return

        # --- New Sorting Category Selected ---
        new_sort_cat = self.sort_categories[sort_cat_index]
        if show_progress:
            progress_msg(
                f"Updating the sorting category of the topics to <{new_sort_cat}>..."
            )
        # Save Old Top Topic to see if it changes.
        old_top_topic = self.topics_tab_cur_topic

        # Update the Current Category Attributes.
        self.cur_cat_index = sort_cat_index
        self.cur_sort_cat = new_sort_cat
        self.cur_cat_topics_info = self.search_engine.topics_info(
            sort_cat=self.cur_sort_cat.lower(), word_num=self.topics_tab_word_num
        )
        # Topics Tab - Get the Top Topic of the new Category.
        top_topic, _, _ = self.cur_cat_topics_info[0]
        self.topics_tab_cur_topic = top_topic

        # Topics Tab - Update Scrollable of Topics.
        self.updateTopicsTabTopicScroll()
        # Topics Tab - Update Scrollable of Documents if the Top Topic Changed.
        if self.topics_tab_cur_topic != old_top_topic:
            self.updateTopicsTabDocScroll()
        # Done.
        if show_progress:
            progress_msg(f"Topics in the Topics Tab sorted by {new_sort_cat}!")

    def updateTopicsTabTopicScroll(self):
        """
        Create or Update the Content of the Topics being displayed in the
        Topics Scrollable in the Topics Tab.
        """
        # Create Layout for the List of Topics.
        top_topics_v_box = QVBoxLayout()
        top_topics_v_box.setSpacing(0)
        top_topics_v_box.setContentsMargins(0, 0, 0, 0)
        top_topics_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Get the Sorting Category and the Topics Info.
        lower_sort_cat = self.cur_sort_cat.lower()
        topics_info = self.cur_cat_topics_info
        # Mark the first Topic of the List.
        is_first = True
        for topic_id, cat_value, description in topics_info:
            topic_info_container = self.topicInfoItem(
                topic_id=topic_id, cat_type=lower_sort_cat, cat_value=cat_value,
                description=description, view_type='vocabulary',
                checkable_type='radio-button', set_checked=is_first,
            )
            top_topics_v_box.addWidget(topic_info_container)
            # Do not check the rest of the topics.
            if is_first:
                is_first = False
        # Create Container for the List of Topics.
        top_topics_container = QWidget()
        top_topics_container.setLayout(top_topics_v_box)

        # Topics Tab - Update Scrollable of Topics.
        self.topics_tab_topics_scroll.setWidget(top_topics_container)

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
        self.updateTopicsTabDocScroll()
        # Done.
        if show_progress:
            progress_msg(f"Documents from <{topic_id}> ready!")

    def updateTopicsTabDocScroll(self):
        """
        Create & Update the Content of the Documents Scrollable in the Topics
        Tab.
        """
        # Get the Topic to display its documents.
        topic_id = self.topics_tab_cur_topic

        # Get the Info about the Documents.
        topic_docs_info = self.search_engine.topic_docs_info(topic_id=topic_id)
        docs_size = len(topic_docs_info)

        # Topic Documents Layout.
        top_docs_v_box = QVBoxLayout()
        top_docs_v_box.setSpacing(0)
        top_docs_v_box.setContentsMargins(0, 0, 0, 0)
        top_docs_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Add items to Top Documents Layout.
        for doc_id, similarity, title, abstract in topic_docs_info:
            red_abstract = abstract[:200] + '...'
            doc_info_container = self.docInfoItem(
                doc_id=doc_id, sim=similarity, title=title, abstract=red_abstract
            )
            top_docs_v_box.addWidget(doc_info_container)
        # Create Container for the Documents Layout.
        top_docs_container = QWidget()
        top_docs_container.setLayout(top_docs_v_box)

        # Update the Label and Scrollable of the Documents.
        self.topics_tab_docs_label.setText(
            f"Documents from {topic_id} ({docs_size} docs):"
        )
        self.topics_tab_docs_scroll.setWidget(top_docs_container)

    def viewVocabulary(self, topic_id: str):
        """
        Show a new Dialog with the Vocabulary of the given Topic 'topic_id'.
        """
        progress_msg("Opening Vocabulary...")
        word_list = self.search_engine.topic_vocab(topic_id)
        self.vocab_window = QVocabDialog(topic_id, word_list)
        self.vocab_window.show()

    def viewDocument(self, doc_id: str):
        """
        Show a Document in the Document Tab.
        """
        progress_msg("View Document is NOT IMPLEMENTED!!!")

    def updateDocTextWidget(self):
        """
        Create or Update the Content of the Text Widget with the current
        Document in the Documents Tab.
        """
        # Get the Title and Abstract.
        doc_id = self.docs_tab_cur_doc
        title = self.search_engine.doc_title(doc_id)
        abstract = self.search_engine.doc_abstract(doc_id)
        # Create Content.
        content = f"""
            <p align='justify'><font face='Times New Roman' size='+2'><u>
            {title}
            </u></font></p>
            <p align='justify'><font size='+1'>{abstract}</font></p>
        """
        # Set the Content inside the TextEdit Widget.
        self.docs_tab_text_widget.setText(content)

    def updateDocsTabTopicsScroll(self):
        """
        Create or Update the Content of the Topics being displayed in the Topics
        Scrollable in the Documents Tab.
        """
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
            topic_info_container = self.topicInfoItem(
                topic_id=topic_id, cat_type='similarity', cat_value=similarity,
                description=description, view_type='topic',
                checkable_type='doc-checkbox', set_checked=is_checked
            )
            # Add to Layout.
            top_topics_v_box.addWidget(topic_info_container)
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
            red_abstract = abstract[:200] + '...'
            doc_info_item = self.docInfoItem(
                doc_id=doc_id, sim=similarity, title=title, abstract=red_abstract
            )
            docs_v_box.addWidget(doc_info_item)
        # Create Container for the Item List.
        docs_container = QWidget()
        docs_container.setLayout(docs_v_box)

        # Update the Document Scrollable inside the Documents Tab.
        self.docs_tab_docs_scroll.setWidget(docs_container)

    def viewFullContent(self, doc_id: str):
        """
        Open a new Dialog to see the content of the Document 'doc_id'.
        """
        progress_msg("Viewing the Full Content NOT IMPLEMENTED!!!")


# Run Application.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(show_progress=True)
    window.show()
    # # Show Window Size after Showing.
    # print(f"Window Width: {window.width()}")
    # print(f"Window Height: {window.height()}")
    sys.exit(app.exec())
