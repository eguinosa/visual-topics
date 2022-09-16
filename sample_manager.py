# Gelin Eguinosa Rosique
# 2022
import json
import random
from os import mkdir
from os.path import isdir, isfile, join

from topic_corpus import TopicCorpus
from corpora_manager import CorporaManager
from extra_funcs import big_number


class SampleManager(TopicCorpus):
    """
    Class to create, store and manage Random Samples of Documents created using
    the documents available in CorporaManager.
    """
    # Class data locations.
    data_folder = 'project_data'
    class_folder = 'sample_manager_files'
    sample_index_suffix = '_sample_index.json'
    sample_info_suffix = '_sample_info.json'

    def __init__(self, corpus: TopicCorpus = None, sample_size=-1,
                 _load_sample=False, _sample_id=''):
        """
        Create a Random Sample of Documents from the provided 'corpus' or the
        default corpus in Corpora Manager. If '_load_sample' is True, then load
        the already saved '_sample_id'.

        Args:
            corpus: TopicCorpus we are going to use to create the random sample
                of documents.
            sample_size: Int with the number of documents we want in the sample.
            _load_sample: Bool indicating if we are going to load an already
                saved random sample of document.
            _sample_id: String with the ID of the sample we are going to load.
        """
        if _load_sample:
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
            # Check the files of the Sample.
            sample_index_file = _sample_id + self.sample_index_suffix
            sample_index_path = join(class_folder_path, sample_index_file)
            if not isfile(sample_index_path):
                raise FileNotFoundError(
                    f"The Index for the Sample <{_sample_id}> does not exist."
                )
            sample_info_file = _sample_id + self.sample_info_suffix
            sample_info_path = join(class_folder_path, sample_info_file)
            if not isfile(sample_info_path):
                raise FileNotFoundError(
                    f"The Info file for the Sample <{_sample_id}> does not exist."
                )
            # Load Index file.
            with open(sample_index_file, 'r') as f:
                sample_doc_ids = json.load(f)
            # Load Info file.
            with open(sample_info_file, 'r') as f:
                sample_info = json.load(f)
            # Load Sample Main Corpus.
            corpus_id = sample_info['corpus_id']
            corpus = CorporaManager(corpus_id=corpus_id)
        else:
            # Load the default corpus in CorporaManager() if none was provided.
            if not corpus:
                corpus = CorporaManager()

            # Get the IDs of the new Sample.
            corpus_ids = corpus.doc_ids
            if sample_size < 0 or len(corpus) <= sample_size:
                sample_doc_ids = corpus_ids
            else:
                sample_doc_ids = random.sample(corpus_ids, sample_size)

        # Save Sample IDs and Corpus Reference.
        self.main_corpus = corpus
        self.sample_doc_ids = sample_doc_ids

    def __len__(self):
        """
        The amount of documents inside the sample.

        Returns: Int with the number of Documents in the Sample.
        """
        result = len(self.sample_doc_ids)
        return result

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

    def save(self, id_prefix=''):
        """
        Save the current Sample to use it later. The number of documents in
        the sample will be the ID to save it, if an 'id_prefix' is provided, it
        will be placed before the number of documents to further identify the
        current random sample.

        Args:
            id_prefix: String with prefix to further identify the current sample.
        """
        # Create Data Folder if it doesn't exist.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        # Create Class Folder if it doesn't exist.
        data_folder_path = join(self.data_folder, self.class_folder)
        if not isdir(data_folder_path):
            mkdir(data_folder_path)

        # Use 'big_number' to create the id of the sample.
        docs_num_str = big_number(len(self.sample_doc_ids)).replace(',', '_')
        # Create filenames.
        if id_prefix:
            sample_index_file = id_prefix + '_' + docs_num_str + self.sample_index_suffix
            sample_info_file = id_prefix + '_' + docs_num_str + self.sample_info_suffix
        else:
            sample_index_file = docs_num_str + self.sample_index_suffix
            sample_info_file = docs_num_str + self.sample_info_suffix

        # Save the Doc's IDs.
        sample_index_path = join(data_folder_path, sample_index_file)
        with open(sample_index_path, 'w') as f:
            json.dump(self.sample_doc_ids, f)
        # Save Original Corpus and Size.
        sample_info = {
            'corpus_id': self.main_corpus.corpus_id,
            'sample_size': len(self.sample_doc_ids),
        }
        sample_info_path = join(data_folder_path, sample_info_file)
        with open(sample_info_path, 'w') as f:
            json.dump(sample_info, f)

    @classmethod
    def available_samples(cls):
        """
        Create a list with the IDs of the available samples.
        """
        raise Exception()
