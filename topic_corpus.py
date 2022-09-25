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

    @abstractmethod
    def corpus_identifier(self):
        """
        Get the ID of the Loaded corpus. It can be used by other classes to
        reload the corpus and its documents, without having to save any other
        information.
        """
        pass

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

    @abstractmethod
    def load_embeddings_dict(self) -> None:
        """
        Load to memory the embeddings Dictionary (around 2GB). Speeds up the
        process to get the embedding of the documents.
        """
        pass

    @abstractmethod
    def unload_embeddings_dict(self) -> None:
        """
        Unload from memory the embeddings Dictionary. Frees up space.
        """
        pass

    def doc_title_abstract(self, doc_id: str):
        """
        Get the title and abstract of the document with the given 'doc_id'.

        Args:
            doc_id: String with the identifier of the document.
        Returns:
            String with the title and abstract of the document.
        """
        title_abstract = self.doc_title(doc_id) + '\n\n' + self.doc_abstract(doc_id)
        return title_abstract

    def corpus_title_abstracts(self):
        """
        Get all the title and abstract of the documents in the corpus.

        Returns: Iterator[Strings] with the title and abstract of the documents.
        """
        for doc_id in self.doc_ids:
            yield self.doc_title_abstract(doc_id)
