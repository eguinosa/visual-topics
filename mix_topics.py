# Gelin Eguinosa Rosique
# 2022

import json
from os import mkdir, listdir
from shutil import rmtree
from os.path import isdir, isfile, join

from base_mix_topics import BaseMixTopics, create_specter_embeds, find_mix_topics
from base_topics import create_topic_words, find_child_embeddings, refresh_topic_ids
from topic_corpus import TopicCorpus
from corpora_manager import CorporaManager
from model_manager import ModelManager
from vocabulary import Vocabulary
from util_funcs import dict_list2ndarray, dict_ndarray2list
from extra_funcs import progress_msg, big_number

# Testing Imports.
import sys
from pprint import pprint
from sample_manager import SampleManager
from time_keeper import TimeKeeper


class MixTopics(BaseMixTopics):
    """
    Class to find the topics inside a corpus using Specter to create the
    embeddings of the Documents and Topics, and another Text Model to create the
    embeddings of the words in the corpus vocabulary, and find the words that
    best describe the topics.
    """
    # Class Data Locations.
    class_folder = 'topic_models'
    model_folder_prefix = 'mix_topics_'
    basic_index_file = 'topic_model_basic_index.json'
    model_index_file = 'topic_model_index.json'
    topic_embeds_docs_file = 'topic_model_embeds_doc_space.json'
    topic_embeds_words_file = 'topic_model_embeds_word_space.json'
    doc_embeds_file = 'topic_model_doc_embeds.json'
    reduced_topics_folder = 'reduced_topic_models'
    reduced_topic_prefix = 'reduced_topic_model_'

    def __init__(self, model_id='', load_model=False, corpus: TopicCorpus = None,
                 vocab_model: ModelManager = None, parallelism=False, show_progress=False):
        """
        Find the topics present in the provided 'corpus' using the Specter
        vector representation of  the documents to find the prominent topics in
        this document space. To find the words that best describe the topics,
        use the 'vocab_model' to create the vector representation of the topics
        and words in the vocabulary, and find the closest words to the topics
        in this Word Vector Space.
          - When creating a new Topic Model, if no 'corpus' is provided uses the
            default corpus in CorporaManager().
          - If no 'vocab_model' is provided, uses the default model in
            ModelManager().
          - If 'load_model' is True, load a saved model using the 'model_id'.
          - If no 'model_id' is provided a new ID is created using the number of
            Documents, the 'vocab_model' name and the number of topics found.

        Args:
             model_id: String with the ID of the Topic Model.
             load_model: Bool indicating if we have to load a saved Topic Model.
             corpus: TopicCorpus representing documents extracted from the
                Cord-19 dataset.
            vocab_model: ModelManager class containing the Text Model used to
                create the embeddings of the topics and words to find the words
                that best describe the topics.
            parallelism: Bool to indicate if we can use parallelism to create
                the topics and create the topics' documents and vocabulary.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Initialize Parent class.
        super().__init__()

        # -- Load Saved Topic Model --
        if load_model:
            # Save Model ID.
            if model_id:
                self.model_id = model_id
            else:
                raise NameError("We need a Topic Model ID to be able to load the model.")

            # Check the Class Data Folder.
            if not isdir(self.class_folder):
                raise NotADirectoryError("The Class Data Folder does not exist.")
            # Check the Model Data Folder.
            model_folder_path = join(self.class_folder, self.model_folder_name)
            if not isdir(model_folder_path):
                raise NotADirectoryError("The Model Folder does not exist.")

            # Load Topic Model's Index for the basic attributes.
            if show_progress:
                progress_msg("Loading Topic Model's Index...")
            model_index_path = join(model_folder_path, self.model_index_file)
            with open(model_index_path, 'r') as f:
                topic_model_index = json.load(f)
            # Get Attributes.
            corpus_id = topic_model_index['corpus_id']
            vocab_model_name = topic_model_index['vocab_model_name']
            topic_docs = topic_model_index['topic_docs']
            topic_words = topic_model_index['topic_words']

            # Load Topic Embeddings.
            if show_progress:
                progress_msg("Loading Topic Embeddings...")
            topic_embeds_docs_path = join(model_folder_path, self.topic_embeds_docs_file)
            if not isfile(topic_embeds_docs_path):
                raise FileNotFoundError(
                    "The File with the Topic Embeddings in the document space "
                    "is not available."
                )
            with open(topic_embeds_docs_path, 'r') as f:
                topic_embeds_docs_index = json.load(f)
            topic_embeds_words_path = join(model_folder_path, self.topic_embeds_words_file)
            if not isfile(topic_embeds_words_path):
                raise FileNotFoundError(
                    "The File with the Topic Embeddings in the vocabulary space "
                    "is not available."
                )
            with open(topic_embeds_words_path, 'r') as f:
                topic_embeds_words_index = json.load(f)
            # Transform Topic Embeddings back to Numpy.ndarray.
            if show_progress:
                progress_msg("Transforming Topic Embeddings to Numpy.ndarray...")
            topic_embeds_docs = dict_list2ndarray(embeds_dict=topic_embeds_docs_index)
            topic_embeds_words = dict_list2ndarray(embeds_dict=topic_embeds_words_index)

            # Load Document Embeddings.
            if show_progress:
                progress_msg("Loading Document Embeddings...")
            doc_embeds_path = join(model_folder_path, self.doc_embeds_file)
            if not isfile(doc_embeds_path):
                raise FileNotFoundError("There is no Document Embeddings file available.")
            with open(doc_embeds_path, 'r') as f:
                doc_embeds_index = json.load(f)
            # Transform Document Embeddings back to Numpy.ndarray.
            if show_progress:
                progress_msg("Transforming Document Embeddings to Numpy.ndarray...")
            doc_embeds = dict_list2ndarray(embeds_dict=doc_embeds_index)

            # Load Topic Model's Vocabulary.
            if show_progress:
                progress_msg("Loading Topic Model's Vocabulary...")
            corpus_vocab = Vocabulary.load(
                topic_dir_path=model_folder_path, show_progress=show_progress
            )

            # Corpus & Text Model Attributes.
            self.corpus_id = corpus_id
            self.vocab_model_name = vocab_model_name
            # Document & Vocabulary Attributes.
            self.doc_embeds = doc_embeds
            self.corpus_vocab = corpus_vocab
            # Topic Attributes.
            self.topic_embeds_docs = topic_embeds_docs
            self.topic_embeds_words = topic_embeds_words
            self.topic_docs = topic_docs
            self.topic_words = topic_words

        # -- Create New Topic Model --
        else:
            # Check if a corpus was provided.
            if not corpus:
                if show_progress:
                    progress_msg("Loading default full corpus to create the Topic Model...")
                corpus = CorporaManager(show_progress=show_progress)
            # Check if the 'vocab_model' was provided.
            if not vocab_model:
                if show_progress:
                    progress_msg("Loading default Text Model to create the Topics Vocabulary...")
                vocab_model = ModelManager(show_progress=show_progress)
            # Save the Corpus ID & Model Name.
            corpus_id = corpus.corpus_identifier()
            vocab_model_name = vocab_model.model_name

            # Get the Embeddings of the Documents.
            if show_progress:
                progress_msg("Getting the Specter Embeddings of the Documents...")
            doc_embeds = create_specter_embeds(
                corpus=corpus, load_full_dict=(len(corpus) > 3_000),
                show_progress=show_progress
            )
            # Use the doc embeddings to find the prominent topics.
            if show_progress:
                progress_msg("Finding the Topics in the Corpus...")
            topic_embeds_docs, topic_embeds_words = find_mix_topics(
                doc_embeds=doc_embeds, corpus=corpus, text_model=vocab_model,
                show_progress=show_progress
            )
            # Group Documents by Topics.
            if show_progress:
                progress_msg("Organizing Documents by topics...")
            topic_docs = find_child_embeddings(
                # parallelism can generate some conflicts with huggingface/tokenizers.
                parent_embeds=topic_embeds_docs, child_embeds=doc_embeds,
                parallelism=parallelism, show_progress=show_progress
            )
            # Create Corpus Vocabulary.
            if show_progress:
                progress_msg("Creating Corpus Vocabulary...")
            corpus_vocab = Vocabulary(
                corpus=corpus, model=vocab_model, show_progress=show_progress
            )
            # Create Topic Words.
            if show_progress:
                progress_msg("Creating Topics' Vocabulary...")
            topic_words = create_topic_words(
                topic_embeds=topic_embeds_words, topic_docs=topic_docs,
                corpus_vocab=corpus_vocab, show_progress=show_progress
            )

            # Corpus & Text Model Attributes.
            self.corpus_id = corpus_id
            self.vocab_model_name = vocab_model_name
            # Document & Vocabulary Attributes.
            self.doc_embeds = doc_embeds
            self.corpus_vocab = corpus_vocab
            # Topic Attributes.
            self.topic_embeds_docs = topic_embeds_docs
            self.topic_embeds_words = topic_embeds_words
            self.topic_docs = topic_docs
            self.topic_words = topic_words
            # Model ID (create a name if none was provided).
            self.model_id = model_id if model_id else self._create_model_id()

        # Create Reduced Topic Model's Attributes with default values.
        self.red_topic_embeds_docs = None
        self.red_topic_embeds_words = None
        self.red_topic_docs = None
        self.red_topic_words = None

    # --------------------------------------------
    # BaseMixTopics Properties
    # --------------------------------------------

    @property
    def base_topic_embeds_docs(self):
        """
        Dictionary with the vector representation of the topics in the same
        vector space as the documents in the corpus.
        """
        return self.topic_embeds_docs

    @property
    def base_topic_embeds_words(self):
        """
        Dictionary with the vector representation of the topics in the same
        vector space as the words in the vocabulary of the corpus.
        """
        return self.topic_embeds_words

    @property
    def base_red_topic_embeds_docs(self):
        """
        Dictionary with the vector representation of the Reduced Topics in the
        same vector space as the words in the vocabulary of the corpus. Used to
        search the words that best represent the topic.
        """
        return self.red_topic_embeds_docs

    # ---------------------------------------------
    # BaseTopics Properties
    # ---------------------------------------------

    @property
    def base_model_id(self) -> str:
        """
        String the ID used to identify the current Topic Model.
        """
        return self.model_id

    @property
    def base_topic_docs(self):
        """
        Dictionary with a list of the IDs of the documents belonging to each
        topic.
        """
        return self.topic_docs

    @property
    def base_cur_topic_docs(self) -> dict:
        """
        Dictionary with the list of Documents (IDs) that belong to each of the
        Reduced Topics.
        """
        return self.red_topic_docs

    @property
    def base_topic_words(self):
        """
        Dictionary with the Topic IDs as keys and the list of words that best
        describe the topics as values.
        """
        return self.topic_words

    @property
    def base_cur_topic_words(self):
        """
        Dictionary the list of words that best describe each of the Reduced
        Topics.
        """
        return self.red_topic_words

    @property
    def base_doc_embeds(self):
        """
        Dictionary with the embeddings of the documents in the corpus.
        """
        return self.doc_embeds

    @property
    def base_corpus_vocab(self):
        """
        Vocabulary class created with the corpus of the Topic Model.
        """
        return self.corpus_vocab

    @property
    def base_class_folder(self) -> str:
        """
        String with the name of the folder where the models of the class will
        be stored.
        """
        return self.class_folder

    @property
    def model_folder_name(self):
        """
        Name of the folder where the Topic Model will be stored.
        """
        model_folder_name = self.model_folder_prefix + self.model_id
        return model_folder_name

    @property
    def base_reduced_folder(self) -> str:
        """
        String with the name of the folder where the Reduced Topic Models will
        be stored.
        """
        return self.reduced_topics_folder

    @property
    def base_reduced_prefix(self) -> str:
        """
        String with the prefix used to create the name of the files used store
        the reduced Topics.
        """
        return self.reduced_topic_prefix

    def reduce_topics(self, new_size: int, parallelism=False, show_progress=False):
        """
        Create a new Hierarchical Topic Model with specified number of topics
        (new_size). The 'new_size' needs to be at least 2, and smaller than the
        current number of topics in the model.

        Args:
            new_size: Int with the desired topic count for the Model.
            parallelism: Bool indicating if we can use multiprocessing or not.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check the topic size requested is valid.
        if not 1 < new_size < self.topic_size:
            if show_progress:
                progress_msg(
                    f"Invalid reduced topic size of {new_size} requested, when "
                    f"the Topic Model only has {self.topic_size} topics."
                )
            # Invalid topic size requested. Reset reduced topics variables.
            self.red_topic_embeds_docs = None
            self.red_topic_embeds_words = None
            self.red_topic_docs = None
            self.red_topic_words = None
            # Exit Function.
            return

        # Get the Topic Embeddings of the Reduced Model with 'new_size' topics.
        if show_progress:
            progress_msg("Creating Reduced Topics...")
        red_topic_embeds_docs, red_topic_embeds_words = self.base_reduce_topics(
            new_size=new_size, parallelism=parallelism, show_progress=show_progress
        )
        # Refresh the Topic IDs (to have the correct max and min ID Number)
        red_topic_embeds_docs = refresh_topic_ids(red_topic_embeds_docs)
        red_topic_embeds_words = refresh_topic_ids(red_topic_embeds_words)
        # Group Documents by the new Reduced Topics.
        if show_progress:
            progress_msg("Organizing documents by the new Reduced Topics...")
        red_topic_docs = find_child_embeddings(
            parent_embeds=red_topic_embeds_docs, child_embeds=self.doc_embeds,
            parallelism=parallelism, show_progress=show_progress
        )
        # Create Topic Words.
        if show_progress:
            progress_msg("Creating the Reduced Topics' Vocabulary...")
        red_topic_words = create_topic_words(
            topic_embeds=red_topic_embeds_words, topic_docs=red_topic_docs,
            corpus_vocab=self.corpus_vocab, show_progress=show_progress
        )

        # Update Reduced Topics Attributes.
        self.red_topic_embeds_docs = red_topic_embeds_docs
        self.red_topic_embeds_words = red_topic_embeds_words
        self.red_topic_docs = red_topic_docs
        self.red_topic_words = red_topic_words

    def refresh_vocabulary(self, show_progress=False):
        """
        Remake the Corpus Vocabulary of the Topic Model using the saved info
        about the Corpus and Text Model used.
        """
        # Create the corpus to remake the Vocabulary.
        doc_ids = list(self.doc_embeds.keys())
        main_corpus = CorporaManager(
            corpus_id=self.corpus_id, show_progress=show_progress
        )
        corpus = SampleManager.load_custom(
            corpus=main_corpus, doc_ids=doc_ids, show_progress=show_progress
        )
        # Create Text Model for the Vocabulary.
        vocab_model = ModelManager(
            model_name=self.vocab_model_name, show_progress=show_progress
        )
        # Create Corpus Vocabulary.
        if show_progress:
            progress_msg("Creating Corpus Vocabulary again...")
        corpus_vocab = Vocabulary(
            corpus=corpus, model=vocab_model, show_progress=show_progress
        )
        # Create Topic Words.
        if show_progress:
            progress_msg("Creating Topics' Vocabulary again...")
        topic_words = create_topic_words(
            topic_embeds=self.topic_embeds_words, topic_docs=self.topic_docs,
            corpus_vocab=corpus_vocab, show_progress=show_progress
        )
        # Update the value of the Attributes.
        self.corpus_vocab = corpus_vocab
        self.topic_words = topic_words
        if show_progress:
            progress_msg("Topic's Attributes Updated!")

        # If the model is saved, update the files of Vocabulary and Topic Words.
        if self.model_saved(model_id=self.model_id):
            # Create path to the Model folder.
            model_folder_path = join(self.class_folder, self.model_folder_name)

            # Update Topic Model's Index file.
            if show_progress:
                progress_msg("Updating Topic Model's Index...")
            topic_model_index = {
                'corpus_id': self.corpus_id,
                'vocab_model_name': self.vocab_model_name,
                'topic_docs': self.topic_docs,
                'topic_words': self.topic_words,
            }
            # Create Path & Save.
            model_index_path = join(model_folder_path, self.model_index_file)
            with open(model_index_path, 'w') as f:
                json.dump(topic_model_index, f)

            # Update Vocabulary files.
            if show_progress:
                progress_msg("Updating Vocabulary files...")
            self.corpus_vocab.save(
                topic_dir_path=model_folder_path, show_progress=show_progress
            )
            # Report all good.
            if show_progress:
                progress_msg("All attribute's files updated!")

    def save(self, show_progress=False):
        """
        Save the Main Topic Model's Attributes, so the model can be loaded later
        using the given or created model ID.

        Args:
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check the Class Data folder.
        if not isdir(self.class_folder):
            mkdir(self.class_folder)
        # Check the Topic Model's Folder path & Create Empty Folder.
        model_folder_path = join(self.class_folder, self.model_folder_name)
        if isdir(model_folder_path):
            # Delete any files in this folder. (Clean everything)
            rmtree(model_folder_path)
        mkdir(model_folder_path)

        # Save Topic Model's Basic Info.
        if show_progress:
            progress_msg("Saving Topic Model's Basic Info...")
        basic_index = {
            'topic_size': self.topic_size,
            'corpus_size': len(self.doc_embeds),
            'corpus_id': self.corpus_id,
            'vocab_model_name': self.vocab_model_name,
            'has_reduced_topics': self.reduced_topics_saved(),
        }
        # Create Path & Save.
        basic_index_path = join(model_folder_path, self.basic_index_file)
        with open(basic_index_path, 'w') as f:
            json.dump(basic_index, f)

        # Save Topic Model's Index with the basic Attributes.
        if show_progress:
            progress_msg("Saving Topic Model's Index...")
        topic_model_index = {
            'corpus_id': self.corpus_id,
            'vocab_model_name': self.vocab_model_name,
            'topic_docs': self.topic_docs,
            'topic_words': self.topic_words,
        }
        # Create Path & Save.
        model_index_path = join(model_folder_path, self.model_index_file)
        with open(model_index_path, 'w') as f:
            json.dump(topic_model_index, f)

        # Transform the Topic Embeddings.
        if show_progress:
            progress_msg("Transforming Topic's Embeddings to List[float]...")
        topic_embeds_docs_index = dict_ndarray2list(embeds_dict=self.topic_embeds_docs)
        topic_embeds_words_index = dict_ndarray2list(embeds_dict=self.topic_embeds_words)
        # Save Topic Embeddings.
        if show_progress:
            progress_msg("Saving Topic's Embeddings...")
        topic_embeds_docs_path = join(model_folder_path, self.topic_embeds_docs_file)
        with open(topic_embeds_docs_path, 'w') as f:
            json.dump(topic_embeds_docs_index, f)
        topic_embeds_words_path = join(model_folder_path, self.topic_embeds_words_file)
        with open(topic_embeds_words_path, 'w') as f:
            json.dump(topic_embeds_words_index, f)

        # Transform Document's Embeddings.
        if show_progress:
            progress_msg("Transforming Document's Embeddings to List[Float]...")
        doc_embeds_index = dict_ndarray2list(self.doc_embeds)
        # Save Document's Embeddings.
        if show_progress:
            progress_msg("Saving Document's Embeddings...")
        doc_embeds_path = join(model_folder_path, self.doc_embeds_file)
        with open(doc_embeds_path, 'w') as f:
            json.dump(doc_embeds_index, f)

        # Save Vocabulary.
        if show_progress:
            progress_msg("Saving Vocabulary...")
        self.corpus_vocab.save(topic_dir_path=model_folder_path, show_progress=show_progress)

    def save_reduced_topics(self, parallelism=False, override=False, show_progress=False):
        """
        Use the base_save_reduced_topics() method to save the Hierarchically
        Reduced Topic Models for the main sizes, and update the information in
        the Basic Index to show that the model now has Reduced Topics.

        Args:
            parallelism: Bool to indicate if we can use multiprocessing to speed
                up the runtime of the method.
            override: Bool indicating if we can delete a previously saved
                Reduced Topics.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check if the model has already saved its reduced topics.
        has_reduced = self.reduced_topics_saved()
        if not override and has_reduced:
            progress_msg(
                "This Topic Model already has its Reduced Topics saved. Set "
                "the 'override' parameter to 'True' to replace them with new "
                "Reduced Topics. "
            )
            # End the method.
            return

        # Use Method from Base Class to save the topics.
        self.base_save_reduced_topics(parallelism, override, show_progress=show_progress)

        # Check if we need to update the Basic Info of the Model.
        if not has_reduced:
            # Update the Basic Info of the Model (It has now a Hierarchy).
            if show_progress:
                progress_msg("Updating Topic Model's Basic Info...")
            basic_index = {
                'topic_size': self.topic_size,
                'corpus_size': len(self.doc_embeds),
                'corpus_id': self.corpus_id,
                'vocab_model_name': self.vocab_model_name,
                'has_reduced_topics': self.reduced_topics_saved(),
            }
            # Create Path & Save.
            model_folder_path = join(self.class_folder, self.model_folder_name)
            basic_index_path = join(model_folder_path, self.basic_index_file)
            with open(basic_index_path, 'w') as f:
                json.dump(basic_index, f)

    @classmethod
    def load(cls, model_id: str, show_progress=False):
        """
        Load a saved Topic Model using its 'model_id'.

        Returns:
            MonoTopics() saved with the given 'model_id'.
        """
        loaded_instance = cls(
            model_id=model_id, load_model=True, show_progress=show_progress
        )
        return loaded_instance

    @classmethod
    def basic_info(cls, model_id: str):
        """
        Load the Basic Info of the Topic Model 'model_id'. Basic Information
        attributes:

        -  topic_size: Int
        -  corpus_size: Int
        -  corpus_id: String
        -  text_model_name: String
        -  has_reduced_topics: Bool

        Args:
            model_id: String with the ID of a saved Topic Model.
        Returns:
            Dictionary with the Basic Info of the Topic Model.
        """
        # Check the Data Folders.
        if not isdir(cls.class_folder):
            raise NotADirectoryError("The Topic Model data folder does not exist.")
        model_folder_name = cls.model_folder_prefix + model_id
        model_folder_path = join(cls.class_folder, model_folder_name)
        if not isdir(model_folder_path):
            raise NotADirectoryError(f"There is not Topic Model saved with the id <{model_id}>.")

        # Load Basic Index.
        basic_index_path = join(model_folder_path, cls.basic_index_file)
        if not isfile(basic_index_path):
            raise FileNotFoundError(f"The Topic Model <{model_id}> has not Basic Index File.")
        with open(basic_index_path, 'r') as f:
            basic_index = json.load(f)
        # Basic Index of the Topic Model.
        return basic_index

    @classmethod
    def model_saved(cls, model_id: str):
        """
        Check if the given 'model_id' string corresponds to a saved Topic Model.

        Args:
            model_id: String with the ID of the Topic Model we want to check.
        Returns:
            Bool indicating if the Topic Model is saved or not.
        """
        # Check model_id.
        if not model_id:
            return False
        # Check Class and Model Folder.
        if not isdir(cls.class_folder):
            return False
        model_folder_name = cls.model_folder_prefix + model_id
        model_folder_path = join(cls.class_folder, model_folder_name)
        if not isdir(model_folder_path):
            return False
        # Check Index and Embeddings Files.
        basic_index_path = join(model_folder_path, cls.basic_index_file)
        if not isfile(basic_index_path):
            return False
        model_index_path = join(model_folder_path, cls.model_index_file)
        if not isfile(model_index_path):
            return False
        topic_embeds_docs_path = join(model_folder_path, cls.topic_embeds_docs_file)
        if not isfile(topic_embeds_docs_path):
            return False
        topic_embeds_words_path = join(model_folder_path, cls.topic_embeds_words_file)
        if not isfile(topic_embeds_words_path):
            return False
        doc_embeds_path = join(model_folder_path, cls.doc_embeds_file)
        if not isfile(doc_embeds_path):
            return False
        # Check Vocabulary.
        if not Vocabulary.vocab_saved(topic_dir_path=model_folder_path):
            return False
        # All Good.
        return True

    @classmethod
    def saved_models(cls):
        """
        Create a list the IDs of the saved Topic Models.

        Returns: List[str] with the IDs.
        """
        # Check the Data Folders.
        if not isdir(cls.class_folder):
            return []

        # Check all the available IDs inside class folder.
        topic_ids = []
        for entry_name in listdir(cls.class_folder):
            # Check we have a valid Model Folder Name.
            entry_path = join(cls.class_folder, entry_name)
            if not isdir(entry_path):
                continue
            if not entry_name.startswith(cls.model_folder_prefix):
                continue

            # Check for the Index Files.
            basic_index_path = join(entry_path, cls.basic_index_file)
            if not isfile(basic_index_path):
                continue
            model_index_path = join(entry_path, cls.model_index_file)
            if not isfile(model_index_path):
                continue
            topic_embeds_docs_path = join(entry_path, cls.topic_embeds_docs_file)
            if not isfile(topic_embeds_docs_path):
                continue
            topic_embeds_words_path = join(entry_path, cls.topic_embeds_words_file)
            if not isfile(topic_embeds_words_path):
                continue
            doc_embeds_path = join(entry_path, cls.doc_embeds_file)
            if not isfile(doc_embeds_path):
                continue
            # Check Vocabulary Files.
            if not Vocabulary.vocab_saved(topic_dir_path=entry_path):
                continue

            # Save Model ID, the folder contains all the main indexes.
            prefix_len = len(cls.model_folder_prefix)
            model_id = entry_name[prefix_len:]
            topic_ids.append(model_id)

        # Sort the Model IDs.
        created_ids = []
        provided_ids = []
        for topic_id in topic_ids:
            if cls._is_created_id(topic_id):
                created_ids.append(topic_id)
            else:
                provided_ids.append(topic_id)
        created_ids.sort(key=lambda x: int(x.split('_docs_')[1][:-7]), reverse=True)
        provided_ids.sort()
        # List with the IDs of all the valid Topic Models.
        topic_ids = created_ids + provided_ids
        return topic_ids

    def _create_model_id(self):
        """
        Create a default ID for the current Topic Model using its corpus size,
        Text Model and the number of topics found.

        Returns: String with the created ID of the topic model.
        """
        mix_name = 'specter_' + self.vocab_model_name
        corpus_size = big_number(len(self.doc_embeds)).replace(',', '_')
        topics_size = big_number(self.topic_size).replace(',', '_')
        new_model_id = (
                mix_name + '_' + corpus_size + '_docs_' + topics_size + '_topics'
        )
        # Example: specter_bert_fast_10_000_docs_100_topics
        return new_model_id

    @classmethod
    def _is_created_id(cls, model_id: str):
        """
        Check if the string 'model_id' was created using the method
        '_create_model_id()'.

        Returns: Bool indicating if the 'model_id' was created by the class.
        """
        # Check has the topics and docs.
        if not model_id.endswith('_topics'):
            return False
        if '_docs_' not in model_id:
            return False
        # Check it has the text model name.
        for vocab_model_name in ModelManager.models_dict.keys():
            mix_name = 'specter_' + vocab_model_name
            if mix_name in model_id:
                stripped_id = model_id.replace(mix_name, '')
                stripped_id = stripped_id.replace('_docs_', '')
                stripped_id = stripped_id.replace('_topics', '')
                stripped_id = stripped_id.replace('_', '')
                # Check it only has numbers after eliminating Topics, Docs and model.
                result = stripped_id.isnumeric()
                return result
        # It doesn't have any Text Model Name.
        return False


if __name__ == '__main__':
    # Record the Runtime of the Program.
    _stopwatch = TimeKeeper()
    # Terminal Parameters.
    _args = sys.argv

    # # Create corpus.
    # # ---------------------------------------------
    # _sample_id = '20_000_docs'
    # print(f"\nLoading the Corpus Sample <{_sample_id}>...")
    # _corpus = SampleManager.load(sample_id=_sample_id, show_progress=True)
    # # ---------------------------------------------
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # # Report amount of papers in the loaded Corpus
    # _paper_count = len(_corpus)
    # print(f"\n{big_number(_paper_count)} documents loaded.")

    # # Load Text Model.
    # _model_id = 'sbert_fast'
    # print(f"\nLoading the model in ModelManager <{_model_id}>...")
    # _text_model = ModelManager(model_name=_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # # Create Topic Model.
    # print(f"\nCreating Topic Model...")
    # _topic_model = MixTopics(
    #     corpus=_corpus, vocab_model=_text_model, show_progress=True
    # )
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # # Report Number of Topics found.
    # print(f"\n{_topic_model.topic_size} topics found.")
    # # ---------------------------------------------
    # # Show Topic by size.
    # print("\nTopics by number of documents:")
    # _topics_sizes = _topic_model.topic_by_size()
    # for _topic_size in _topics_sizes:
    #     print(_topic_size)
    # # ---------------------------------------------
    # # Topics' Vocabulary
    # top_n = 15
    # print(f"\nTop {top_n} words per topic:")
    # _topics_words = _topic_model.topics_top_words(n=top_n)
    # for _topic_id, _topic_words in _topics_words.items():
    #     print(f"\n-----> {_topic_id}:")
    #     pprint(_topic_words)

    # # -- Test Saving Model --
    # print(f"\nSaving Topic Model <{_topic_model.model_id}>...")
    # _topic_model.save(show_progress=True)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # -- Test Loading Topic Model --
    _loading_id = 'specter_sbert_fast_20_000_docs_119_topics'  # _topic_model.model_id
    print(f"\nLoading Topic Model with ID <{_loading_id}>...")
    _loaded_model = MixTopics.load(model_id=_loading_id, show_progress=True)
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")
    # ---------------------------------------------
    # Show Loaded Topics.
    print(f"\nThe Loaded Topic Model has {_loaded_model.topic_size} topics.")
    # ---------------------------------------------
    # Show Topics.
    print("\nTopic by number of documents (Loaded Model):")
    _topics_sizes = _loaded_model.topic_by_size()
    for _topic_size in _topics_sizes:
        print(_topic_size)
    # ---------------------------------------------
    # # Show Topics' Words.
    # top_n = 15
    # print(f"\nTop {top_n} words per topic:")
    # _topics_words = _loaded_model.topics_top_words(n=top_n)
    # for _topic_id, _topic_words in _topics_words.items():
    #     print(f"\n-----> {_topic_id}:")
    #     pprint(_topic_words)

    # # -- Update the Vocabulary of the Topic Model --
    # print("\nCreating again the Vocabulary of the model...")
    # _loaded_model.refresh_vocabulary(show_progress=True)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # -- Topics' Words using PWI --
    # top_n = 15
    # print(f"\nTop {top_n} words per topic:")
    # for _topic_id, _size in _loaded_model.topic_by_size():
    #     print(f"\n{_topic_id} ({big_number(_size)} docs):")
    #     _sim_words = _loaded_model.top_words_topic(_topic_id, top_n, 'cos_sim')
    #     print("Top Words by Cosine Similarity:")
    #     pprint(_sim_words)
    #     _pwi_words = _loaded_model.top_words_topic(_topic_id, top_n, 'pwi_exact')
    #     print("Top Words by PWI-exact:")
    #     pprint(_pwi_words)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # # --Test Creating Hierarchically Reduced Topics--
    # # Save the Hierarchically Reduced Topic Models.
    # print(f"\nSaving Reduced Topics for the Model <{_loaded_model.model_id}>...")
    # _loaded_model.save_reduced_topics(parallelism=True, override=False, show_progress=True)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # # -- Create Hierarchically Reduced Topics --
    _new_size = 20
    print(f"\nCreating Reduced Model with {_new_size} topics...")
    _loaded_model.reduce_topics(new_size=_new_size, parallelism=False, show_progress=True)
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")
    # # ---------------------------------------------
    # Show Reduced Topics.
    print("\nReduced Topics (by number of docs):")
    _red_topics_sizes = _loaded_model.cur_topic_by_size()
    for _red_topic_size in _red_topics_sizes:
        print(_red_topic_size)
    # # ---------------------------------------------
    # # Show Topic Words.
    # _top_n = 15
    # print(f"\nTop {_top_n} words per reduced topic:")
    # _red_topic_words = _loaded_model.red_topics_top_words(n=_top_n)
    # for _red_topic_id, _red_words in _red_topic_words.items():
    #     print(f"\n-----> {_red_topic_id}:")
    #     pprint(_red_words)

    # # -- Topics' Words in Reduced Topics using PWI --
    top_n = 15
    print(f"\nTop {top_n} words per topic:")
    for _topic_id, _size in _loaded_model.cur_topic_by_size():
        print(f"\n{_topic_id} ({big_number(_size)} docs):")
        _sim_words = _loaded_model.top_words_cur_topic(_topic_id, top_n, 'cos_sim')
        print("Top Words by Cosine Similarity:")
        pprint(_sim_words)
        _pwi_words = _loaded_model.top_words_cur_topic(_topic_id, top_n, 'pwi_exact')
        print("Top Words by PWI-exact:")
        pprint(_pwi_words)

    # -- Show the Topic Model Descriptive Value (PWI) --
    _num = 20
    print(f"\nReduced Topic Model Descriptive Value with {_num} words:")
    _pwi_tf_idf = _loaded_model.cur_model_pwi(word_num=_num, pwi_type='tf-idf')
    _pwi_exact = _loaded_model.cur_model_pwi(word_num=_num, pwi_type='exact')
    print(f"  PWI-tf-idf: {_pwi_tf_idf}")
    print(f"  PWI-exact: {_pwi_exact}")

    # -- Show Saved Models --
    _saved_topic_models = MixTopics.saved_models()
    if _saved_topic_models:
        print("\nSaved Topic Models:")
    else:
        print("\nNo Topic Models Saved.")
    for _model_id in _saved_topic_models:
        print(f"  -> {_model_id}")

    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
