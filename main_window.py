# Gelin Eguinosa Rosique
# 2022

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QTabWidget, QLayout, QScrollArea, QLineEdit, QPushButton, QFrame,
    QButtonGroup, QCheckBox, QRadioButton
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

        # Gather Data Related with the Current Topic Model.
        topic_size = str(search_engine.topic_size)
        supported_sizes = [str(x) for x in search_engine.supported_model_sizes()]
        cur_size_index = supported_sizes.index(topic_size)
        # Topics Sorting Categories.
        sort_categories = ['Size', 'PWI-tf-idf', 'PWI-exact']
        cur_sort_cat = 'size'
        cur_cat_index = 0

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

        # Topic Size - Combo Boxes
        self.search_tab_size_combo = None
        self.topics_tab_size_combo = None
        self.docs_tab_size_combo = None
        # Topics - Scrollable Areas
        self.search_tab_topics_scroll = None
        self.topics_tab_topics_scroll = None
        self.docs_tab_topics_scroll = None
        # Windows Tabs - Information.
        self.search_tab_index = 0
        self.topic_tab_index = 1
        self.doc_tab_index = 2
        # Topics Tab - Information.
        self.topics_tab_cur_topic = ''
        self.topics_tab_button_group = QButtonGroup()
        self.topics_tab_topics_scroll = None
        self.topics_tab_docs_label = None
        self.topics_tab_docs_scroll = None

        # Menu Bar Actions.
        self.quit_act = None

        # Create App UI.
        self.initializeUI(show_progress=show_progress)

    def initializeUI(self, show_progress=False):
        """
        Set up the Application's GUI
        """
        # Create Window Size and Name.
        self.setMinimumSize(1000, 800)
        # self.setGeometry(100, 80, 1300, 900)
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
        tab_bar.addTab(docs_tab, "Documents")
        tab_bar.setCurrentIndex(self.topic_tab_index)

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
        topics_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        topics_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        topics_scroll_area.setWidget(top_topics_container)
        topics_scroll_area.setWidgetResizable(True)
        # -- Topics Final Layout --
        topics_layout = QVBoxLayout()
        topics_layout.addWidget(size_container)
        topics_layout.addWidget(top_topics_label)
        topics_layout.addWidget(topics_scroll_area)  # , 0, Qt.AlignmentFlag.AlignHCenter)
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
        docs_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        docs_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
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
        topics_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        topics_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        topics_scroll_area.setWidgetResizable(True)
        self.topics_tab_topics_scroll = topics_scroll_area
        # Fill Scrollable with the Topics Information.
        self.changeTopicSorting(
            sort_cat_index=self.cur_cat_index, is_initialization=True,
            show_progress=True
        )
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
        docs_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        docs_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        docs_scroll_area.setWidgetResizable(True)
        self.topics_tab_docs_scroll = docs_scroll_area
        # Update Documents Label and Scrollable with the current topic.
        self.newTopicSelected(
            checked=True, topic_id=self.topics_tab_cur_topic, show_progress=True
        )
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
        if show_progress:
            progress_msg("Creating Documents Tab...")
        return QWidget()

    def topicInfoItem(
            self, topic_id: str, cat_type: str, cat_value, description='',
            use_view_button=True, checkable_type='checkbox', set_checked=False
    ):
        """
        Build a List Item for the given 'topic_id'. The Item built depends on
        the information provided, if description is empty, the use the search
        engine to load it.

        The variable 'checkable_type' determines the type of check button will
        be used for the topics, either QCheckboxes (checkbox) or QRadioButton
        (radio-button).
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
        if checkable_type == 'checkbox':
            header_checkable = QCheckBox(topic_header)
            if set_checked:
                header_checkable.setChecked(True)
            header_checkable.toggled.connect(
                lambda checked, x=topic_id: self.updatedTopicCheckbox(checked, x)
            )
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
        if use_view_button:
            view_button = QPushButton("View Topic")
            view_button.clicked.connect(
                lambda x=topic_id: self.viewTopic(topic_id=x)
            )
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

    def changeTopicSorting(
            self, sort_cat_index: int, is_initialization=False, show_progress=False
    ):
        """
        Change the sorting category use to rank the topics in the Topics Tab.
        """
        # Check if we have a new sorting category.
        if sort_cat_index == self.cur_cat_index and not is_initialization:
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

        # Update the Current Category Attributes.
        self.cur_cat_index = sort_cat_index
        self.cur_sort_cat = new_sort_cat

        # Create Layout for the List of Topics.
        top_topics_v_box = QVBoxLayout()
        top_topics_v_box.setSpacing(0)
        top_topics_v_box.setContentsMargins(0, 0, 0, 0)
        top_topics_v_box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Add Items to Top Topics Layout using the Default Topics by Size.
        lower_sort_cat = new_sort_cat.lower()
        topics_info = self.search_engine.topics_info(
            sort_cat=lower_sort_cat, word_num=15
        )
        # Check the top topic.
        is_first = True
        for topic_id, cat_value, description in topics_info:
            topic_info_container = self.topicInfoItem(
                topic_id=topic_id, cat_type=lower_sort_cat, cat_value=cat_value,
                description=description, use_view_button=False,
                checkable_type='radio-button', set_checked=is_first,
            )
            top_topics_v_box.addWidget(topic_info_container)
            # Do not check the rest of the topics.
            if is_first:
                self.topics_tab_cur_topic = topic_id
                is_first = False
        # Create Container for the List of Topics.
        top_topics_container = QWidget()
        top_topics_container.setLayout(top_topics_v_box)

        # Topics Tab - Update Scrollable of Topics.
        self.topics_tab_topics_scroll.setWidget(top_topics_container)
        if show_progress:
            progress_msg(f"Topics in the Topics Tab sorted by {new_sort_cat}!")

    def viewTopic(self, topic_id: str):
        """
        Open the Topics Tab to view the Topic 'topic_id'.
        """
        progress_msg("View Topic is NOT YET IMPLEMENTED!!!")

    def updatedTopicCheckbox(self, checked: bool, topic_id: str):
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

    def newTopicSelected(self, checked: bool, topic_id: str, show_progress=False):
        """
        Update the Information being displayed in the Topics Tab, using the new
        Topic selected.
        """
        # Check if this the new enabled Topic.
        if not checked:
            # No need to do any changes to the UI.
            return

        # --- Display the Documents of the New Topic ---
        if show_progress:
            progress_msg(f"Creating Layout for the Documents from <{topic_id}>...")
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
        # Done.
        if show_progress:
            progress_msg(f"Documents from <{topic_id}> ready!")


# Run Application.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(show_progress=True)
    window.show()
    sys.exit(app.exec())
