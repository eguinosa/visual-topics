# Gelin Eguinosa Rosique
# 2022

from PyQt6.QtCore import pyqtSignal, QThread, QObject

from ir_system import IRSystem


class QModelWorker(QThread):
    """
    Class to Work on the Topic Model of the IR System without freezing the
    UI of the app.
    """
    # Class Signals.
    task_done = pyqtSignal()

    def __init__(
            self, search_engine: IRSystem, new_model_name='', new_topic_size='',
            parent_widget: QObject = None, show_progress=False
    ):
        """
        Initialize the class and its attributes.
        """
        # Check that we have a task to do.
        if not new_model_name and not new_topic_size:
            raise AttributeError("This class needs something to do.")
        # Initialize the base class.
        super().__init__(parent_widget)
        # Save class attributes.
        self.search_engine = search_engine
        self.new_model_name = new_model_name
        self.new_topic_size = new_topic_size
        self.show_progress = show_progress

    def run(self):
        """
        Start a separate thread to do some work on the Topic Model from the
        IR System without freezing the app.
        """
        # Change the Topic Model in the IR System.
        if self.new_model_name:
            self.search_engine.update_model(
                new_model=self.new_model_name, show_progress=self.show_progress
            )
        # Change the Topic Size of the Model.
        if self.new_topic_size:
            self.search_engine.update_topic_size(
                new_size=self.new_topic_size, show_progress=self.show_progress
            )
        # Done working on the Topic Model.
        # noinspection PyUnresolvedReferences
        self.task_done.emit()
