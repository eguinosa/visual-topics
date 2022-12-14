# Gelin Eguinosa Rosique
# 2022

import sys
import json
import random
from os import mkdir, listdir
from os.path import isdir, isfile, join

from topic_corpus import TopicCorpus
from corpora_manager import CorporaManager
from extra_funcs import big_number, progress_msg

# Testing Imports.
from time_keeper import TimeKeeper


class SampleManager(TopicCorpus):
    """
    Class to create, store and manage Random Samples of Documents created using
    the documents available in CorporaManager.
    """
    # Class data locations.
    data_folder = 'project_data'
    class_folder = 'sample_manager_files'
    sample_index_suffix = '_sample_index.json'

    def __init__(
            self, corpus: TopicCorpus = None, sample_size=-1, _load_sample=False,
            _sample_id='', _load_custom=False, _ids_list: list = None,
            show_progress=False
    ):
        """
        Create a Random Sample of Documents from the provided 'corpus' or the
        default corpus in Corpora Manager.

        If '_load_sample' is True, then load the already saved '_sample_id'.

        If '_load_custom' is True, then create a custom sample with the given
        'corpus' as the main corpus, and 'ids_list' as the IDs of the documents
        in the sample.

        Args:
            corpus: TopicCorpus we are going to use to create the random sample
                of documents.
            sample_size: Int with the number of documents we want in the sample.
            _load_sample: Bool indicating if we are going to load an already
                saved random sample of document.
            _sample_id: String with the ID of the sample we are going to load.
            _load_custom: Indicates if we are loading a custom Sample, created
                with a given '_id_list' with the Doc IDs we want on the sample.
            _ids_list: List[str] with the IDs of the Documents that will be in
                the sample.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Initialize Parent Class.
        super().__init__()

        # Check if we have to load or create the sample.
        if _load_sample:
            if show_progress:
                progress_msg(f"Loading Saved Sample <{_sample_id}>...")
            # Check the Data folders.
            if not isdir(self.data_folder):
                raise FileNotFoundError(
                    f"The Data Folder for the Sample <{_sample_id}> does not exist."
                )
            class_folder_path = join(self.data_folder, self.class_folder)
            if not isdir(class_folder_path):
                raise FileNotFoundError(
                    f"The Class Folder for the Sample <{_sample_id}> does not exist."
                )
            # Load Index file.
            sample_index_file = _sample_id + self.sample_index_suffix
            sample_index_path = join(class_folder_path, sample_index_file)
            if not isfile(sample_index_path):
                raise FileNotFoundError(
                    f"The Index for the Sample <{_sample_id}> does not exist."
                )
            with open(sample_index_path, 'r') as f:
                sample_index = json.load(f)
            # Load Class Attributes.
            corpus_id = sample_index['corpus_id']
            sample_doc_ids = sample_index['sample_doc_ids']
            corpus = CorporaManager(corpus_id=corpus_id, show_progress=show_progress)
        elif _load_custom:
            if not corpus:
                raise ValueError(
                    "To create a custom sample, the Corpus can't be empty."
                )
            if not _ids_list:
                raise ValueError(
                    "We need the list of the Doc's IDs to create the custom "
                    "sample."
                )
            # Create the custom Sample with the corpus and Doc's IDs.
            if show_progress:
                progress_msg("Creating a custom Sample with provided IDs.")
            sample_doc_ids = _ids_list
        else:
            if show_progress:
                progress_msg("Creating a new Sample of documents...")
            # Load the default corpus in CorporaManager() if none was provided.
            if not corpus:
                corpus = CorporaManager(show_progress=show_progress)
            # Get the IDs of the new Sample.
            corpus_ids = corpus.doc_ids
            if sample_size < 0 or len(corpus) <= sample_size:
                sample_doc_ids = corpus_ids
            else:
                sample_doc_ids = random.sample(corpus_ids, sample_size)

        # Save Sample IDs and Corpus Reference.
        self.main_corpus = corpus
        self.sample_doc_ids = sample_doc_ids
        if show_progress:
            docs_number = big_number(len(sample_doc_ids))
            progress_msg(f"Sample with {docs_number} documents ready.")

    def __len__(self):
        """
        The amount of documents inside the sample.

        Returns: Int with the number of Documents in the Sample.
        """
        result = len(self.sample_doc_ids)
        return result

    def corpus_identifier(self):
        """
        Get the ID of the Loaded corpus. It can be used by other classes to
        reload the corpus and its documents, without having to save any other
        information.

        Returns: String with the ID of the current corpus.
        """
        return self.main_corpus.corpus_id

    @property
    def doc_ids(self):
        """
        Get the Ids of the documents in the current sample.

        Returns:
            List[str] with the IDs of the documents.
        """
        return self.sample_doc_ids

    def doc_title(self, doc_id: str):
        """
        Get the title of the document 'doc_id'.

        Args:
            doc_id: String with the ID of the document.

        Returns:
            String with the title of the document.
        """
        return self.main_corpus.doc_title(doc_id)

    def doc_abstract(self, doc_id: str):
        """
        Get the abstract of the document 'doc_id'.

        Args:
            doc_id: String with the ID of the document.

        Returns:
            String with the abstract of the document.
        """
        return self.main_corpus.doc_abstract(doc_id)

    def doc_body_text(self, doc_id: str):
        """
        Get the body text of the document 'doc_id'. Return an empty string ('')
        if the document has only title & abstract.

        Args:
            doc_id: String with the ID of the document.

        Returns:
            String with the text in the body of the document.
        """
        return self.main_corpus.doc_body_text(doc_id)

    def doc_specter_embed(self, doc_id: str):
        """
        Get the Specter Embedding of the Document.

        Args:
            doc_id: String with the ID of the document.

        Returns: List[float] with the Specter embedding of the Document.
        """
        return self.main_corpus.doc_specter_embed(doc_id)

    def load_embeddings_dict(self):
        """
        Load to memory the embeddings Dictionary (around 2GB). Speeds up the
        process to get the embedding of the documents.
        """
        self.main_corpus.load_embeddings_dict()

    def unload_embeddings_dict(self):
        """
        Unload from memory the embeddings Dictionary. Frees up space.
        """
        self.main_corpus.unload_embeddings_dict()

    def save(self, sample_id=''):
        """
        Save the current Sample to use it later. If the 'sample_id' is not
        provided, the number of documents in the sample will be used as the ID.

        Args:
            sample_id: String with an ID to identify the current sample.
        """
        # Create Data Folder if it doesn't exist.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        # Create Class Folder if it doesn't exist.
        data_folder_path = join(self.data_folder, self.class_folder)
        if not isdir(data_folder_path):
            mkdir(data_folder_path)

        # Create filenames.
        if sample_id:
            sample_index_file = sample_id + self.sample_index_suffix
        else:
            # Use the number of documents to create the id of the sample.
            docs_num_str = big_number(len(self.sample_doc_ids)).replace(',', '_')
            custom_id = docs_num_str + '_docs'
            sample_index_file = custom_id + self.sample_index_suffix

        # Save Sample Index.
        sample_index = {
            'corpus_id': self.main_corpus.corpus_id,
            'sample_doc_ids': self.sample_doc_ids
        }
        sample_index_path = join(data_folder_path, sample_index_file)
        with open(sample_index_path, 'w') as f:
            json.dump(sample_index, f)

    @classmethod
    def load(cls, sample_id: str, show_progress=False):
        """
        Load a previously saved sample using its ID 'sample_id'.

        Args:
            sample_id: String with the ID of the sample we want to load.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        Returns:
            SampleManager corresponding to the given 'sample_id'.
        """
        if not sample_id:
            raise FileNotFoundError("We need an ID to load a Sample.")
        sample = cls(_load_sample=True,  _sample_id=sample_id, show_progress=show_progress)
        return sample

    @classmethod
    def load_custom(cls, corpus: CorporaManager, doc_ids: list, show_progress=False):
        """
        Create a Sample with the 'corpus' and 'doc_ids' provided.
        """
        sample = cls(
            corpus=corpus, _load_custom=True, _ids_list=doc_ids, show_progress=show_progress
        )
        return sample

    @classmethod
    def saved_samples(cls):
        """
        Create a list with the IDs of the available samples.

        Returns: List[str] with the sample's IDs.
        """
        # Check the Data & Class folders.
        if not isdir(cls.data_folder):
            return []
        data_folder_path = join(cls.data_folder, cls.class_folder)
        if not isdir(data_folder_path):
            return []

        # Create default list for the IDs.
        sample_ids = []
        # Get the elements inside the class folder that represent a Sample.
        for file_name in listdir(data_folder_path):
            new_file_path = join(data_folder_path, file_name)
            # Only examine files.
            if not isfile(new_file_path):
                continue
            # See if we have an Index file.
            if file_name.endswith(cls.sample_index_suffix):
                id_len = len(file_name) - len(cls.sample_index_suffix)
                sample_id = file_name[:id_len]
                sample_ids.append(sample_id)

        # Sort the Sample IDs.
        custom_ids = []
        provided_ids = []
        for sample_id in sample_ids:
            if cls.is_custom_id(sample_id):
                custom_ids.append(sample_id)
            else:
                provided_ids.append(sample_id)
        custom_ids.sort(key=lambda x: int(x[:-5]))
        provided_ids.sort()
        # List of the Sample IDs found.
        sample_ids = custom_ids + provided_ids
        return sample_ids

    @classmethod
    def is_custom_id(cls, sample_id: str):
        """
        Determine if the 'sample_id' was created by the class, or it was created
        by a user.
        """
        if not sample_id.endswith('_docs'):
            return False
        number_str = sample_id[:-5].replace('_', '')
        if number_str.isnumeric():
            return True
        return False


if __name__ == '__main__':
    # Record the runtime of the Program.
    _stopwatch = TimeKeeper()
    # Get the Console arguments.
    _args = sys.argv

    # # -- Create a Sample --
    # _size = 10
    # print(f"\nCreating a Sample of {_size} documents...")
    # _sample = SampleManager(sample_size=_size)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")
    #
    # # Show Doc IDs.
    # print(f"\nSample size: {big_number(len(_sample.doc_ids))}")
    # print("Sample IDs:")
    # print(_sample.doc_ids)

    # -- Save Sample --
    # # ------------------------------------------------
    # # _old_id = 'old_sample'
    # # print(f"\nSaving Sample with ID <{_old_id}>...")
    # # _sample.save(sample_id=_old_id)
    # # ------------------------------------------------
    # print("\nSaving Sample...")
    # _sample.save()
    # # ------------------------------------------------
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # # -- Create New Sample --
    # _new_size = 10
    # print(f"\nCreating a new Sample of {_new_size} documents...")
    # _sample = SampleManager(sample_size=_new_size)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")
    # ------------------------------------------------
    # # Show New Doc Ids.
    # print("\nNew Sample IDs:")
    # print(_sample.doc_ids)
    # ------------------------------------------------
    # # Save the New Sample.
    # # the_new_id = 'new_sample'
    # # print(f"\nSaving New Sample with ID <{the_new_id}>...")
    # print("\nSaving New Sample with no ID...")
    # _sample.save()
    # print("Sample Saved.")
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # # -- Load Old sample --
    # print(f"\nLoading old Sample <{_old_id}>...")
    # _old_sample = SampleManager.load(sample_id=_old_id)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")
    # ------------------------------------------------
    # # Show Sample IDs.
    # print("\nOld Sample IDs:")
    # print(_old_sample.doc_ids)

    # # -- Create a Custom Sample --
    # print("\nLoading a Custom Sample...")
    # _source_corpus = _sample.main_corpus
    # _source_ids = _sample.doc_ids
    # _custom_sample = SampleManager.load_custom(
    #     corpus=_source_corpus, doc_ids=_source_ids, show_progress=True
    # )
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")
    # # --------------------------------------------------
    # print(f"\n{big_number(len(_custom_sample))} documents loaded.")
    # # --------------------------------------------------
    # for _doc_id in _custom_sample.doc_ids:
    #     # Show Doc ID and Title.
    #     print(f"\nDoc <{_doc_id}>")
    #     print(f"Title: {_custom_sample.doc_title(_doc_id)}")

    # Print the Available Samples.
    print("\nSaved Samples:")
    _saved_samples = SampleManager.saved_samples()
    for an_id in _saved_samples:
        print(f"  -> {an_id}")

    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
