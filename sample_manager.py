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
    sample_info_suffix = '_sample_info.json'

    def __init__(self, corpus: TopicCorpus = None, sample_size=-1,
                 _load_sample=False, _sample_id='', show_progress=False):
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
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
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
            with open(sample_index_path, 'r') as f:
                sample_doc_ids = json.load(f)
            # Load Info file.
            with open(sample_info_path, 'r') as f:
                sample_info = json.load(f)
            # Load Sample Main Corpus.
            corpus_id = sample_info['corpus_id']
            corpus = CorporaManager(corpus_id=corpus_id, show_progress=show_progress)
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
            sample_info_file = sample_id + self.sample_info_suffix
        else:
            # Use the number of documents to create the id of the sample.
            docs_num_str = big_number(len(self.sample_doc_ids)).replace(',', '_')
            custom_id = docs_num_str + '_docs'
            sample_index_file = custom_id + self.sample_index_suffix
            sample_info_file = custom_id + self.sample_info_suffix

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
    def load(cls, sample_id: str):
        """
        Load a previously saved sample using its ID 'sample_id'.

        Returns: SampleManager corresponding to the given 'sample_id'.
        """
        if not sample_id:
            raise FileNotFoundError("We need an ID to load a Sample.")
        sample = cls(_load_sample=True,  _sample_id=sample_id)
        return sample

    @classmethod
    def available_samples(cls):
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
                # Check we haven't found this ID before.
                if sample_id not in sample_ids:
                    sample_ids.append(sample_id)
            # See if we have an Info file.
            elif file_name.endswith(cls.sample_info_suffix):
                id_len = len(file_name) - len(cls.sample_info_suffix)
                sample_id = file_name[:id_len]
                # Check we haven't found this ID before.
                if sample_id not in sample_ids:
                    sample_ids.append(sample_id)

        # List the sample IDs found.
        return sample_ids


if __name__ == '__main__':
    # Record the runtime of the Program.
    stopwatch = TimeKeeper()
    # Get the Console arguments.
    args = sys.argv

    # # Create a Sample.
    # the_size = 10
    # print(f"\nCreating a Sample of {the_size} documents...")
    # the_sample = SampleManager(sample_size=the_size)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Show Doc IDs.
    # print("\nSample IDs:")
    # print(the_sample.doc_ids)
    #
    # # Save Sample.
    # the_old_id = 'old_sample'
    # print(f"\nSaving Sample with ID <{the_old_id}>...")
    # the_sample.save(sample_id=the_old_id)
    # print("Sample Saved.")
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Create another Sample.
    # the_new_size = 10
    # print(f"\nCreating a new Sample of {the_new_size} documents...")
    # the_sample = SampleManager(sample_size=the_new_size)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Show New Doc Ids.
    # print("\nNew Sample IDs:")
    # print(the_sample.doc_ids)
    #
    # # Save the New Sample.
    # # the_new_id = 'new_sample'
    # # print(f"\nSaving New Sample with ID <{the_new_id}>...")
    # print("\nSaving New Sample with no ID...")
    # the_sample.save()
    # print("Sample Saved.")
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Load Old sample.
    # print(f"\nLoading old Sample <{the_old_id}>...")
    # the_old_sample = SampleManager.load(sample_id=the_old_id)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Show Sample IDs.
    # print("\nOld Sample IDs:")
    # print(the_old_sample.doc_ids)

    # Print the Available Samples.
    print("\nAvailable Samples:")
    the_saved_samples = SampleManager.available_samples()
    for an_id in the_saved_samples:
        print(f"  -> {an_id}")

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
