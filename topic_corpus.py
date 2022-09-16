# Gelin Eguinosa Rosique
# 2022

from abc import ABC, abstractmethod


class TopicCorpus(ABC):
    """
    Class to represent a Corpus used to create a Topic Model.
    """

    def __len__(self):
        """
        The Number of Documents in the Corpus.
        """
        return len(self.doc_ids)

    @property
    @abstractmethod
    def doc_ids(self):
        """
        Get the list of the IDs of the Documents in the Corpus.
        """
        pass

    @abstractmethod
    def doc_title(self, doc_id: str):
        """
        Get the title of the document 'doc_id'.
        """
        pass

    @abstractmethod
    def doc_abstract(self, doc_id: str):
        """
        Get the abstract of the document 'doc_id'.
        """
        pass

    @abstractmethod
    def doc_body_text(self, doc_id: str):
        """
        Get the body text of the document 'doc_id'. Return an empty string ('')
        if the document has only title & abstract.
        """
        pass

    @abstractmethod
    def doc_specter_embed(self, doc_id: str):
        """
        Get the specter embedding of the document 'doc_id'.
        """
        pass
