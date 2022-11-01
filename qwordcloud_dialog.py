# Gelin Eguinosa Rosique
# 2022

import sys
from wordcloud import WordCloud
from multidict import MultiDict
from PyQt6.QtCore import Qt

from matplotlib.backends.qt_compat import QtWidgets
# noinspection PyUnresolvedReferences
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure


class QWordCloudDialog(QtWidgets.QDialog):
    """
    Class to show in a Dialog the vocabulary of a Topic using Word Clouds.
    """

    def __init__(
            self, topic_id: str, words_sims_list: list,
            parent_widget: QtWidgets.QWidget = None
    ):
        """
        Initialize class and attributes.
        """
        # Initialize the base class.
        super().__init__(parent_widget)
        self.setModal(False)
        # Save the class attributes.
        self.topic_id = topic_id
        self.words_sims_list = words_sims_list
        # Create the UI.
        self.initializeUI()

    def initializeUI(self):
        """
        Set up the Application's GUI.
        """
        self.setMinimumSize(650, 450)
        self.setWindowTitle(f"Word Cloud - {self.topic_id}")
        self.setUpMainWindow()

    def setUpMainWindow(self):
        """
        Create and arrange widgets in the main window.
        """
        # Text for the Header.
        word_count = len(self.words_sims_list)
        header_text = f"""
            <p align='left'><font face='Times New Roman' size='+1'>
            Top {word_count} words from {self.topic_id}:
            </font></p>
        """
        # Header Label.
        header_label = QtWidgets.QLabel(header_text)
        # Done Button.
        done_button = QtWidgets.QPushButton('Done')
        done_button.clicked.connect(self.close)
        # Image.
        wordcloud_canvas = FigureCanvas(Figure(figsize=(8, 4), tight_layout=True))

        # Create Main Layout.
        main_v_box = QtWidgets.QVBoxLayout()
        main_v_box.addWidget(header_label)
        main_v_box.addWidget(wordcloud_canvas)
        main_v_box.addWidget(NavigationToolbar(wordcloud_canvas, self))
        main_v_box.addWidget(done_button, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(main_v_box)

        # --- Create Word Cloud ---
        # Change the Word Similarities to Int (x 1000).
        new_words_sims = [
            (word, round(sim * 1000))
            for word, sim in self.words_sims_list
        ]
        # Generate Word Cloud using the words and their frequency.
        freq_dict = MultiDict(new_words_sims)
        word_cloud = WordCloud(
            width=1600, height=800, collocations=False, background_color='white'
        )
        # word_cloud = WordCloud(background_color='white')
        # noinspection PyTypeChecker
        word_cloud.generate_from_frequencies(freq_dict)
        # Create Word Cloud Image.
        wordcloud_ax = wordcloud_canvas.figure.subplots()
        wordcloud_ax.imshow(word_cloud, interpolation='bilinear')
        wordcloud_ax.axis('off')


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    # Example Word Sims.
    custom_word_sims = [
        ('cuba', 0.90),
        ('fiesta', 0.80),
        ('locura', 0.70),
        ('vida', 0.60),
        ('viajar', 0.50),
        ('tiempo', 0.45),
        ('felicidad', 0.40),
        ('camino', 0.35),
        ('tiempo', 0.30),
        ('siesta', 0.25),
        ('libertad', 0.20)
    ]
    # Create Dialog.
    app = QWordCloudDialog(topic_id='Topic_03', words_sims_list=custom_word_sims)
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
