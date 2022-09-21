# Gelin Eguinosa
# 2022

import json
from os import mkdir
from shutil import rmtree
from os.path import isdir, isfile, join

from base_topics import (
    create_docs_embeds, find_topics, find_child_embeddings, best_midway_sizes
)
from topic_corpus import TopicCorpus
from corpora_manager import CorporaManager
from model_manager import ModelManager
from vocabulary import Vocabulary
from util_funcs import dict_list2ndarray, dict_ndarray2list
from extra_funcs import progress_msg, big_number


class MonoTopics:
    """
    Class to find the topics inside a corpus using the same Text Model for the
    representation of the documents, the topics, and the words in the vocabulary.
    """
    # Class Data Locations.
    class_folder = 'topic_models'
    model_folder_prefix = 'topics_'
    basic_index_file = 'topic_model_basic_index.json'
    model_index_file = 'topic_model_index.json'
    doc_embeds_file = 'topic_model_doc_embeds.json'
    topic_embeds_file = 'topic_model_topic_embeds.json'
    reduced_topics_folder = 'reduced_topic_models'
    reduced_topic_prefix = 'reduced_topic_model_'

    def __init__(self, corpus: TopicCorpus = None, model: ModelManager = None,
                 _used_saved=False, _model_id='', parallelism=False,
                 show_progress=False):
        """
        Find the topics present in the provided 'corpus' using 'model' to create
        the vector representation of the topics, documents and words.
          - When creating a new Topic Model, if no 'corpus' is provided uses the
            default corpus in CorporaManager().
          - If no 'model' is provided, uses the default model in ModelManager().
          - If '_used_saved' is True, loads a saved model using the '_model_id'.

        Args:
            corpus: TopicCorpus representing documents extracted from the
                Cord-19 dataset.
            model: ModelManager class containing a Text Model to create the
                embeddings of the documents and words in the corpus.
            _used_saved: Bool indicating if we have to load a saved Topic Model.
            _model_id: String with the ID of the saved Topic Model.
            parallelism: Bool to indicate if we can use parallelism to create
                the topics and create the topics' documents and vocabulary.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # -- Load Saved Topic Model --
        if _used_saved:
            # Check the Class Data Folder.
            if not isdir(self.class_folder):
                raise NotADirectoryError("The Class Data Folder does not exist.")
            # Check the Model Data Folder.
            model_folder_name = self.model_folder_prefix + _model_id
            model_folder_path = join(self.class_folder, model_folder_name)

            # Load Topic Model's Index for the basic attributes.
            if show_progress:
                progress_msg("Loading Topic Model's Index...")
            model_index_path = join(model_folder_path, self.model_index_file)
            if not isfile(model_index_path):
                raise FileNotFoundError("The Topic Model's Index file is not available.")
            with open(model_index_path, 'r') as f:
                topic_model_index = json.load(f)
            # Get the Attributes.
            topic_size = topic_model_index['topic_size']
            corpus_id = topic_model_index['corpus_id']
            text_model_name = topic_model_index['text_model_name']
            topic_docs = topic_model_index['topic_docs']

            # Load Topic Embeddings.
            if show_progress:
                progress_msg("Loading Topic Embeddings...")
            topic_index_path = join(model_folder_path, self.topic_embeds_file)
            if not isfile(topic_index_path):
                raise FileNotFoundError("There is no Topic Embeddings file available.")
            with open(topic_index_path, 'r') as f:
                topic_embeds_index = json.load(f)
            # Transform Topic Embeddings back to Numpy.ndarray.
            if show_progress:
                progress_msg("Transforming Topic Embeddings to Numpy.ndarray...")
            topic_embeds = dict_list2ndarray(topic_embeds_index, show_progress=show_progress)

            # Load Document Embeddings.
            if show_progress:
                progress_msg("Loading Document Embeddings...")
            doc_embeds_path = join(model_folder_path, self.doc_embeds_file)
            if not isdir(doc_embeds_path):
                raise FileNotFoundError("There is no Document Embeddings file available.")
            with open(doc_embeds_path, 'r') as f:
                doc_embeds_index = json.load(f)
            # Transform Document Embeddings back to Numpy.ndarray.
            if show_progress:
                progress_msg("Transforming Document Embeddings to Numpy.ndarray...")
            doc_embeds = dict_list2ndarray(doc_embeds_index, show_progress=show_progress)

            # Load Topic Model's Vocabulary.
            if show_progress:
                progress_msg("Loading Topic Model's Vocabulary...")
            corpus_vocab = Vocabulary.load(model_folder_path, show_progress=show_progress)
        # -- Create New Topic Model --
        else:
            # Check if a corpus was provided.
            if not corpus:
                if show_progress:
                    progress_msg("Loading default full corpus to create the Topic Model...")
                corpus = CorporaManager(show_progress=show_progress)
            # Check if a Text Model was provided.
            if not model:
                if show_progress:
                    progress_msg("Loading default Text Model to create the Topic Model...")
                model = ModelManager(show_progress=show_progress)
            # Save the Corpus ID & Model Name.
            corpus_id = corpus.corpus_identifier()
            text_model_name = model.model_name

            # Get the embeddings of the documents.
            if show_progress:
                progress_msg("Creating Document Embeddings...")
            doc_embeds = create_docs_embeds(corpus, model, show_progress=show_progress)
            # Use the embeddings of the documents to find the prominent topics.
            if show_progress:
                progress_msg("Finding the Topics...")
            topic_embeds = find_topics(
                doc_embeds_list=doc_embeds.values(), show_progress=show_progress
            )
            # Report the number of topics found.
            topic_size = len(topic_embeds)
            if show_progress:
                progress_msg(f"{topic_size} topics found.")
            # Group Documents by Topics.
            if show_progress:
                progress_msg("Organizing documents by topics...")
            topic_docs = find_child_embeddings(
                # parallelism can generate some conflicts with huggingface/tokenizers.
                parent_embeds=topic_embeds, child_embeds=doc_embeds,
                parallelism=parallelism, show_progress=show_progress
            )
            # Create Corpus Vocabulary.
            if show_progress:
                progress_msg("Creating Corpus Vocabulary...")
            corpus_vocab = Vocabulary(corpus, model, show_progress=show_progress)

        # Save Corpus and Model Info.
        self.corpus_id = corpus_id
        self.text_model_name = text_model_name
        # Save Corpus Documents and Vocabulary.
        self.doc_embeds = doc_embeds
        self.corpus_vocab = corpus_vocab
        # Save Topic Model's Attributes.
        self.topic_size = topic_size
        self.topic_embeds = topic_embeds
        self.topic_docs = topic_docs
        # Create Reduced Topic Model's Attributes with default values.
        self.reduced_topic = False  # Bool indicating if we have reduced topics.
        self.new_topic_size = 0
        self.new_topic_embeds = None
        self.new_topic_docs = None

    def save(self, model_id='', show_progress=False):
        """
        Save the Main Topic Model's Attributes, so the model can be loaded later
        using the given or created 'model_id'. If no 'model_id' is provided
        a new ID is created using the number of Documents, the Text Model, and
        the number topics found.

        Args:
            model_id: String with the ID we want to use to save the Topic Model.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check if we have to create a Topic Model.
        if not model_id:
            model_id = self.create_model_id()

        # Check the Class Data folder.
        if not isdir(self.class_folder):
            mkdir(self.class_folder)
        # Clean the Topic Model's Folder path.
        model_folder_name = self.model_folder_prefix + model_id
        model_folder_path = join(self.class_folder, model_folder_name)
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
            'text_model_type': self.text_model_name,
            'has_reduced_topics': self.reduced_topics_saved(model_id),
        }
        # Create Path & Save.
        basic_index_path = join(model_folder_path, self.basic_index_file)
        with open(basic_index_path, 'w') as f:
            json.dump(basic_index, f)

        # Save Topic Model's Index with the basic Attributes.
        if show_progress:
            progress_msg("Saving Topic Model's Index...")
        topic_model_index = {
            'topic_size': self.topic_size,
            'corpus_id': self.corpus_id,
            'text_model_name': self.text_model_name,
            'topic_docs': self.topic_docs
        }
        # Create Index path and Save.
        model_index_path = join(model_folder_path, self.model_index_file)
        with open(model_index_path, 'w') as f:
            json.dump(topic_model_index, f)

        # Transform the Topic Embeddings.
        if show_progress:
            progress_msg("Transforming Topic's Embeddings to List[float]...")
        topic_embeds_index = dict_ndarray2list(self.topic_embeds, show_progress=show_progress)
        # Save Topics Embeddings.
        if show_progress:
            progress_msg("Saving Topic's Embeddings...")
        topic_embeds_path = join(model_folder_path, self.topic_embeds_file)
        with open(topic_embeds_path, 'w') as f:
            json.dump(topic_embeds_index, f)

        # Transform Document's Embeddings.
        if show_progress:
            progress_msg("Transforming Document's Embeddings to List[Float]...")
        doc_embeds_index = dict_ndarray2list(self.doc_embeds, show_progress=show_progress)
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

    def create_model_id(self):
        """
        Create a default ID for the current Topic Model using its corpus size,
        Text Model and the number of topics found.

        Returns: String with the created ID of the topic model.
        """
        num_docs = big_number(len(self.doc_embeds)).replace(',', '_')
        model_name = self.text_model_name
        topics_size = big_number(self.topic_size).replace(',', '_')
        new_model_id = model_name + '_' + num_docs + '_docs_' + topics_size + 'topics'
        # Example: bert_fast_10_000_docs_100_topics
        return new_model_id

    def reduced_topics_saved(self, model_id: str):
        """
        Check if the given Topic Model created and saved the Hierarchically
        Reduced Topics.

        Args:
            model_id: ID used to save the Topic Model.
        Returns:
            Bool showing if the Reduced Topic Models were saved.
        """
        # Check Model ID.
        if not model_id:
            return False
        # Check the Model's Folders.
        if not isdir(self.class_folder):
            return False
        model_folder_name = self.model_folder_prefix + model_id
        model_folder_path = join(self.class_folder, model_folder_name)
        if not isdir(model_folder_path):
            return False
        reduced_folder_path = join(model_folder_path, self.reduced_topics_folder)
        if not isdir(reduced_folder_path):
            return False

        # Check that all the Main Reduced Topic Models were saved.
        main_sizes = best_midway_sizes(self.topic_size)
        for topic_size in main_sizes:
            # Check the file for the Reduced Model with the current size.
            reduced_topic_file = self.reduced_topic_prefix + str(topic_size) + '.json'
            reduced_topic_path = join(reduced_folder_path, reduced_topic_file)
            if not isfile(reduced_topic_path):
                return False

        # All The Files were created correctly.
        return True

#     def generate_new_topics(self, number_topics: int, parallelism=False, show_progress=False):
#         """
#         Create a new Hierarchical Topic Model with specified number of topics
#         (num_topics). The 'num_topics' need to be at least 2 topics, and be
#         smaller than the original number of topics found.

#         Args:
#             number_topics: The desired topic count for the new Topic Model.
#             parallelism: Bool indicating if we have to use multiprocessing or not.
#             show_progress: Bool representing whether we show the progress of
#                 the function or not.
#         """
#         # Check the number of topics requested is valid.
#         if not 1 < number_topics < self.num_topics:
#             # Invalid topic size requested. Reset reduced topics variables.
#             self.new_topics = False
#             self.new_topic_embeds = None
#             self.new_num_topics = None
#             self.new_topic_docs = None
#             self.new_topic_words = None
#             self.topics_hierarchy = None
#             # Progress.
#             if show_progress:
#                 progress_msg("Invalid number of topics requested. No"
#                              " hierarchical topic reduction performed.")
#             # Exit function.
#             return

#         # Check if we can use a previously calculated Reduced Topic Model.
#         use_saved_model = self.reduced_topics_saved()
#         if not use_saved_model:
#             # Initialize New Topics Variables.
#             current_num_topics = self.num_topics
#             new_topic_embeds = self.topic_embeds.copy()
#             new_topic_sizes = dict([(topic_id, len(self.topic_docs[topic_id]))
#                                     for topic_id in self.topic_docs.keys()])
#         else:
#             # Get the closest viable Reduced Topic Size.
#             main_sizes = best_midway_sizes(self.num_topics)
#             usable_sizes = [size for size in main_sizes if size >= number_topics]
#             optimal_size = min(usable_sizes)
#             # Upload The Reduced Topic Model.
#             if show_progress:
#                 progress_msg(f"Loading Reduced Topic Model with {optimal_size} topics...")
#             model_folder_name = self.model_folder_prefix + self.model_id
#             reduced_topic_file = self.reduced_topic_prefix + str(optimal_size) + '.json'
#             reduced_topic_path = join(self.data_folder, self.class_data_folder,
#                                       model_folder_name, self.reduced_topics_folder,
#                                       reduced_topic_file)
#             with open(reduced_topic_path, 'r') as f:
#                 reduced_topic_index = json.load(f)
#             # Get the Reduced Topic's Sizes.
#             loaded_topic_sizes = reduced_topic_index['topic_sizes']
#             new_topic_sizes = dict([(int(key), size)
#                                     for key, size in loaded_topic_sizes.items()])
#             # Get the Reduced Topic's Embeddings.
#             json_topic_embeds = reduced_topic_index['topic_embeds']
#             new_topic_embeds = {}
#             for topic_id, topic_embed in json_topic_embeds.items():
#                 new_topic_embeds[int(topic_id)] = np.array(topic_embed)
#             # Update Current Topic Size.
#             current_num_topics = len(new_topic_embeds)

#         # Perform topic reduction until we get the desired number of topics.
#         while number_topics < current_num_topics:
#             # Reduce the number of topics by 1.
#             if show_progress:
#                 progress_msg(f"Reducing from {current_num_topics} to {current_num_topics - 1} topics...")
#             result_tuple = self._reduce_topic_size(ref_topic_embeds=new_topic_embeds,
#                                                    topic_sizes=new_topic_sizes,
#                                                    parallelism=parallelism,
#                                                    show_progress=show_progress)
#             new_topic_embeds, new_topic_sizes = result_tuple
#             # Update Current Number of Topics.
#             current_num_topics = len(new_topic_embeds)

#         # Progress - Done with the reduction of Topics.
#         if show_progress:
#             progress_msg(f"No need to reduce the current {current_num_topics} topics.")

#         # Update New Topics' Attributes.
#         self.new_topics = True
#         self.new_num_topics = current_num_topics
#         # Reset IDs of the New Topics.
#         self.new_topic_embeds = dict([(new_id, topic_embed)
#                                       for new_id, topic_embed
#                                       in enumerate(new_topic_embeds.values())])
#         # Assign Words and Documents to the New Topics.
#         if show_progress:
#             progress_msg("Organizing documents using the New Topics...")
#         self.new_topic_docs = find_child_embeddings(self.new_topic_embeds,
#                                                     self.doc_embeds,
#                                                     parallelism=parallelism,
#                                                     show_progress=show_progress)
#         if show_progress:
#             progress_msg("Creating the vocabulary for the New Topics...")
#         self.new_topic_words = find_child_embeddings(self.new_topic_embeds,
#                                                      self.word_embeds,
#                                                      parallelism=parallelism,
#                                                      show_progress=show_progress)
#         # Assign Original Topics to the New Topics.
#         if show_progress:
#             progress_msg("Assigning original topics to the New topics...")
#         self.topics_hierarchy = find_child_embeddings(self.new_topic_embeds,
#                                                       self.topic_embeds,
#                                                       parallelism=parallelism,
#                                                       show_progress=show_progress)

#     def _reduce_topic_size(self, ref_topic_embeds: dict, topic_sizes: dict,
#                            parallelism=False, show_progress=False):
#         """
#         Reduce the provided Topics in 'ref_topic_embeds' by 1, mixing the
#         smallest topic with its closest neighbor.

#         Args:
#             ref_topic_embeds: Dictionary containing the embeddings of the topics
#                 we are going to reduce. This dictionary is treated as a
#                 reference and will be modified to store the new reduced topics.
#             topic_sizes: Dictionary containing the current size of the topics we
#                 are reducing.
#             parallelism: Bool to indicate if we have to use the multiprocessing
#                 version of this function.
#             show_progress: A Bool representing whether we show the progress of
#                 the function or not.

#         Returns:
#             Tuple with 'ref_topic_embeds' dictionary  and a new 'topic_sizes'
#                 dictionary containing the updated embeddings and sizes
#                 respectively for the new Topics.
#         """
#         # Get the smallest topic and its info.
#         new_topics_list = list(ref_topic_embeds.keys())
#         min_topic_id = min(new_topics_list, key=lambda x: len(self.topic_docs[x]))
#         min_embed = ref_topic_embeds[min_topic_id]

#         # Delete Smallest Topic.
#         del ref_topic_embeds[min_topic_id]
#         # Get the closest topic to the small topic.
#         close_topic_id, _ = closest_vector(min_embed, ref_topic_embeds)
#         close_embed = ref_topic_embeds[close_topic_id]

#         # Merge the embedding of the topics.
#         min_size = topic_sizes[min_topic_id]
#         close_size = topic_sizes[close_topic_id]
#         total_size = min_size + close_size
#         merged_topic_embed = (min_size * min_embed + close_size * close_embed) / total_size

#         # Update embedding of the closest topic.
#         ref_topic_embeds[close_topic_id] = merged_topic_embed
#         # Get the new topic sizes.
#         if show_progress:
#             progress_msg(f"Creating sizes for the new {len(ref_topic_embeds)} topics...")
#         new_topic_sizes = self._topic_document_count(ref_topic_embeds,
#                                                      parallelism=parallelism,
#                                                      show_progress=show_progress)
#         # New Dictionaries with embeds and sizes.
#         return ref_topic_embeds, new_topic_sizes

#     def top_topics(self, show_originals=False):
#         """
#         Make a sorted list with the topics organized by the amount of documents
#         they represent.

#         Args:
#             show_originals: Bool to indicate if we need to use the original
#                 topics even if we already have found New Topics.

#         Returns:
#             A list of tuples with the topics' ID and their document count.
#         """
#         # Choose between the original topics or the hierarchical reduced.
#         if show_originals or not self.new_topics:
#             topic_docs_source = self.topic_docs
#         else:
#             topic_docs_source = self.new_topic_docs

#         # Form list of topics with their size.
#         topic_docs = [(topic_id, len(docs_list))
#                       for topic_id, docs_list in topic_docs_source.items()]
#         # Sort by size.
#         topic_docs.sort(key=lambda count: count[1], reverse=True)
#         return topic_docs

#     def _topic_document_count(self, topic_embeds_dict: dict, parallelism=False,
#                               show_progress=False):
#         """
#         Given a dictionary with the embeddings of a group of topics, count the
#         number of documents assign to each of the topics in the given dictionary.

#         Args:
#             topic_embeds_dict: Dictionary with the topic IDs as keys and the
#                 embeddings of the topics as values.
#             parallelism: Bool to indicate if we have to use the multiprocessing
#                 version of this function.
#             show_progress: A Bool representing whether we show the progress of
#                 the function or not.

#         Returns:
#             Dictionary containing the topic IDs as keys and the number of
#                 documents belonging to each one in the current corpus.
#         """
#         # Check if multiprocessing was requested, and we have enough topics.
#         # parallel_min = int(2 * PEAK_SIZE / MAX_CORES)  # I made a formula that game me this number (?)
#         parallel_min = 37  # This is the number when more cores are faster.
#         if parallelism and len(topic_embeds_dict) > parallel_min:
#             return self._document_count_parallel(topic_embeds_dict, show_progress)

#         # Check we have at least a topic.
#         if len(topic_embeds_dict) == 0:
#             return {}

#         # Progress Variables.
#         count = 0
#         total = len(self.doc_embeds)
#         # Iterate through the documents and their embeddings.
#         topic_docs_count = {}
#         for doc_id, doc_embed in self.doc_embeds.items():
#             # Find the closest topic to the current document.
#             topic_id, _ = closest_vector(doc_embed, topic_embeds_dict)
#             # Check if we have found this topic before.
#             if topic_id in topic_docs_count:
#                 topic_docs_count[topic_id] += 1
#             else:
#                 topic_docs_count[topic_id] = 1
#             # Show Progress:
#             if show_progress:
#                 count += 1
#                 progress_bar(count, total)

#         # The document count per each topic.
#         return topic_docs_count

#     def _document_count_parallel(self, topic_embeds_dict: dict, show_progress=False):
#         """
#         Version of _topic_document_count() using MultiProcessing.

#         Given a dictionary with the embeddings of a group of topics, count the
#         number of documents assign to each of the topics in the given dictionary.

#         Args:
#             topic_embeds_dict: Dictionary with the topic IDs as keys and the
#                 embeddings of the topics as values.
#             show_progress: A Bool representing whether we show the progress of
#                 the function or not.

#         Returns:
#             Dictionary containing the topic IDs as keys and the number of
#                 documents belonging to each one in the current corpus.
#         """
#         # Check we have at least a topic.
#         if len(topic_embeds_dict) == 0:
#             return {}

#         # Determine the number of cores.
#         optimal_cores = min(multiprocessing.cpu_count(), MAX_CORES)
#         efficiency_mult = min(float(1), len(topic_embeds_dict) / PEAK_SIZE)
#         core_count = max(2, int(efficiency_mult * optimal_cores))
#         # Number of instructions processed in one batch.
#         chunk_size = max(1, len(self.doc_embeds) // 100)
#         # chunk_size = max(1, len(self.doc_embeds) // (PARALLEL_MULT * core_count))

#         # Dictionary for the Topic-Docs count.
#         topic_docs_count = defaultdict(int)
#         # Create parameter tuples for _custom_closest_vector().
#         tuple_params = [(doc_id, doc_embed, topic_embeds_dict)
#                         for doc_id, doc_embed in self.doc_embeds.items()]
#         with multiprocessing.Pool(processes=core_count) as pool:
#             # Pool map() vs imap() depending on if we have to report progress.
#             if show_progress:
#                 # Report Parallelization.
#                 progress_msg(f"Using Parallelization <{core_count} cores>")
#                 # Progress Variables.
#                 count = 0
#                 total = len(self.doc_embeds)
#                 # Iterate through the results to update the process.
#                 for doc_id, topic_id, _ in pool.imap(_custom_closest_vector, tuple_params, chunksize=chunk_size):
#                     topic_docs_count[topic_id] += 1
#                     # Progress.
#                     count += 1
#                     progress_bar(count, total)
#             else:
#                 # Process all the parameters at once (faster).
#                 tuple_results = pool.map(_custom_closest_vector, tuple_params, chunksize=chunk_size)
#                 for doc_id, topic_id, _ in tuple_results:
#                     topic_docs_count[topic_id] += 1

#         # The document count per each topic.
#         topic_docs_count = dict(topic_docs_count)
#         return topic_docs_count

#     def save_reduced_topics(self, parallelism=False, show_progress=False):
#         """
#         Create a list of basic topic sizes between 2 and the size of the current
#         Topic Model, to create and save the Hierarchical Topic Models of this
#         Model with these sizes, so when we create a new Hierarchical Topic Model
#         we can do it faster, only having to start reducing the Topic sizes from
#         the closest basic topic size.

#         The saved topic sizes will be in the range of 2-1000, with different
#         steps depending on the Topic Size range.
#           - Step of  5 between  2 and 30.
#           - Step of 10 between 30 and 100.
#           - Step of 25 between 100 and 300.
#           - Step of 50 between 300 and 1000.

#         Args:
#             parallelism: Bool to indicate if we have to use the multiprocessing
#                 version of this function.
#             show_progress:  A Bool representing whether we show the progress of
#                 the function or not.
#         """
#         # Check we can create a Reduced Topic Model.
#         if self.num_topics <= 2:
#             return

#         # Check the class' data folders.
#         if not isdir(self.data_folder):
#             mkdir(self.data_folder)
#         class_folder_path = join(self.data_folder, self.class_data_folder)
#         if not isdir(class_folder_path):
#             mkdir(class_folder_path)
#         model_folder_name = self.model_folder_prefix + self.model_id
#         model_folder_path = join(class_folder_path, model_folder_name)
#         if not isdir(model_folder_path):
#             mkdir(model_folder_path)
#         # Check if there is an already Saved Hierarchy.
#         reduced_folder_path = join(model_folder_path, self.reduced_topics_folder)
#         if isdir(reduced_folder_path):
#             # Remove the previously saved Hierarchy.
#             rmtree(reduced_folder_path)
#         # Create Empty Folder to store the hierarchically reduced topics.
#         mkdir(reduced_folder_path)

#         # Get a Set with the Reduced Topic Sizes that we have to save.
#         main_sizes = best_midway_sizes(self.num_topics)

#         # Initialize Topic Reduction dictionaries.
#         current_num_topics = self.num_topics
#         new_topic_embeds = self.topic_embeds.copy()
#         new_topic_sizes = dict([(topic_id, len(self.topic_docs[topic_id]))
#                                 for topic_id in self.topic_docs.keys()])

#         # Start Reducing Topics and Saving the Main Topic Sizes.
#         if show_progress:
#             progress_msg("Saving Main Hierarchically Reduced Topic Models...")
#         while current_num_topics > 2:
#             # Reduce number of topics by 1.
#             if show_progress:
#                 progress_msg(f"Reducing from {current_num_topics} to {current_num_topics - 1} topics...")
#             result_tuple = self._reduce_topic_size(ref_topic_embeds=new_topic_embeds,
#                                                    topic_sizes=new_topic_sizes,
#                                                    parallelism=parallelism,
#                                                    show_progress=show_progress)
#             new_topic_embeds, new_topic_sizes = result_tuple
#             # Update current number of topics.
#             current_num_topics = len(new_topic_embeds)

#             # Check if we need to save the current embeddings and sizes.
#             if current_num_topics in main_sizes:
#                 if show_progress:
#                     progress_msg("<<Main Topic Found>>")
#                     progress_msg(f"Saving Reduced Topic Model with {current_num_topics} topics...")
#                 # Transform Embeddings to lists.
#                 json_topic_embeds = {}
#                 for topic_id, topic_embed in new_topic_embeds.items():
#                     json_topic_embeds[topic_id] = topic_embed.tolist()
#                 # Create Dict with embeddings and sizes.
#                 reduced_topic_index = {
#                     'topic_embeds': json_topic_embeds,
#                     'topic_sizes': new_topic_sizes,
#                 }
#                 # Save the index of the reduced topic.
#                 reduced_topic_file = (self.reduced_topic_prefix
#                                       + str(current_num_topics) + '.json')
#                 reduced_topic_path = join(reduced_folder_path, reduced_topic_file)
#                 with open(reduced_topic_path, 'w') as f:
#                     json.dump(reduced_topic_index, f)
#                 # Progress.
#                 if show_progress:
#                     progress_msg("<<Saved>>")

#         # Update the Basic Info of the Model (It has now a Hierarchy).
#         self.save_basic_info()

#     @classmethod
#     def load(cls, model_id: str = None, show_progress=False):
#         """
#         Load a previously saved Topic Model.

#         Args:
#             model_id: String with the ID we are using to identify the topic
#                 model.
#             show_progress: A Bool representing whether we show the progress of
#                 the function or not.
#         Returns:
#             An instance of the TopicModel class.
#         """
#         return cls(model_id=model_id, used_saved=True, show_progress=show_progress)

#     @classmethod
#     def basic_info(cls, model_id: str = None):
#         """
#         Load the Basic Model Info of the Topic Model 'model_id'.

#         Args:
#             model_id: String with the ID of the Topic Model.

#         Returns:
#             Dictionary with the 'model_type', 'num_topics', 'corpus_size' and
#                 the 'topics_hierarchy' bool of the Topic Model.
#         """
#         # If no 'model_id' is provided load the Default Model.
#         if not model_id:
#             model_id = cls.default_model_id

#         # Check the existence of the data folders.
#         if not isdir(cls.data_folder):
#             raise FileNotFoundError("There is no Data Folder available.")
#         class_folder_path = join(cls.data_folder, cls.class_data_folder)
#         if not isdir(class_folder_path):
#             raise FileNotFoundError("There is no Class Data Folder available.")
#         model_folder_name = cls.model_folder_prefix + model_id
#         model_folder_path = join(class_folder_path, model_folder_name)
#         if not isdir(model_folder_path):
#             raise FileNotFoundError(f"The Topic Model <{model_id}> has no Data Folder.")

#         # Load the Basic Index.
#         basic_index_path = join(model_folder_path, cls.basic_index_file)
#         if not isfile(basic_index_path):
#             raise FileNotFoundError(f"The Topic Model <{model_id}> has not Basic Index File.")
#         with open(basic_index_path, 'r') as f:
#             basic_index = json.load(f)

#         # The Basic Index of the Topic Model.
#         return basic_index

#     @classmethod
#     def saved_topic_models(cls):
#         """
#         Create a list with the IDs of the saved Topic Models.

#         Returns: List[String] with the IDs of the saved Topic Models.
#         """
#         # Check the Data Folders.
#         if not isdir(cls.data_folder):
#             return []
#         class_folder_path = join(cls.data_folder, cls.class_data_folder)
#         if not isdir(class_folder_path):
#             return []

#         # Check all the available IDs inside class folder.
#         topic_ids = []
#         for entry_name in listdir(class_folder_path):
#             entry_path = join(class_folder_path, entry_name)
#             # Check we have a valid Model Folder Name.
#             if not isdir(entry_path):
#                 continue
#             if not entry_name.startswith(cls.model_folder_prefix):
#                 continue

#             # Check for the Index Files.
#             basic_index_path = join(entry_path, cls.basic_index_file)
#             if not isfile(basic_index_path):
#                 continue
#             model_index_path = join(entry_path, cls.model_index_file)
#             if not isfile(model_index_path):
#                 continue
#             word_embeds_path = join(entry_path, cls.word_embeds_file)
#             if not isfile(word_embeds_path):
#                 continue
#             doc_embeds_path = join(entry_path, cls.doc_embeds_file)
#             if not isfile(doc_embeds_path):
#                 continue
#             topic_embeds_path = join(entry_path, cls.topic_embeds_file)
#             if not isfile(topic_embeds_path):
#                 continue

#             # Save Model ID, the folder contains all the main indexes.
#             model_id = entry_name.replace(cls.model_folder_prefix, '', 1)
#             topic_ids.append(model_id)

#         # List with the IDs of all the valid Topic Models.
#         return topic_ids


# if __name__ == '__main__':
#     # Record the Runtime of the Program
#     stopwatch = TimeKeeper()

#     # -- Load Corpus --
#     # # Load Random Sample to use a limited amount of papers from CORD-19.
#     # test_size = 500
#     # print(f"\nLoading Random Sample of {big_number(test_size)} documents...")
#     # sample = RandomSample(paper_type='medium', sample_size=test_size, show_progress=True)
#     # sample = RandomSample.load(show_progress=True)

#     # Load RandomSample() saved with an id.
#     sample_id = '5000_docs'
#     print(f"\nLoading Saved Random Sample <{sample_id}>...")
#     sample = RandomSample.load(sample_id=sample_id, show_progress=True)
#     # ---------------------------------------------
#     # # Use CORD-19 Dataset
#     # print("\nLoading the CORD-19 Dataset...")
#     # sample = PapersCord19(show_progress=True)
#     print("Done.")
#     print(f"[{stopwatch.formatted_runtime()}]")

#     # -- Report amount of papers in the loaded Corpus --
#     papers_count = len(sample.papers_cord_uids())
#     print(f"\n{big_number(papers_count)} papers loaded.")

#     # -- Load Document Model --
#     # Use BERT Document Model.
#     print("\nLoading Bert Model...")
#     # bert_name = 'all-MiniLM-L12-v2'
#     bert_name = 'paraphrase-MiniLM-L3-v2'  # Fastest model.
#     my_model = BertCord19(model_name=bert_name, show_progress=True)
#     # ---------------------------------------------
#     # # Use Specter Document Model.
#     # print("\nLoading Specter model...")
#     # my_model = SpecterManager(show_progress=True)
#     # ---------------------------------------------
#     # Use Doc2Vec Model trained with Cord-19 papers.
#     # print("\nLoading Doc2Vec model of the Cord-19 Dataset...")
#     # my_model = Doc2VecCord19.load('cord19_dataset', show_progress=True)
#     print("Done.")
#     print(f"[{stopwatch.formatted_runtime()}]")

#     # -- Load Topic Model --
#     # the_model_id = 'testing_' + my_model.model_type()
#     the_model_id = f'test_{my_model.model_type()}_{sample_id}'
#     # ---------------------------------------------
#     # Creating Topic Model.
#     print(f"\nCreating Topic Model with ID <{the_model_id}>...")
#     the_topic_model = TopicModel(corpus=sample, doc_model=my_model, only_title_abstract=True,
#                                  model_id=the_model_id, show_progress=True)
#     # ---------------------------------------------
#     # print(f"Saving Topic Model with ID <{the_topic_model.model_id}>...")
#     # # new_model_id = the_model_id
#     # new_model_id = the_model_id + f"_{the_topic_model.num_topics}topics"
#     # the_topic_model.save(model_id=new_model_id, show_progress=True)
#     # ---------------------------------------------
#     # # Loading Saved Topic Model.
#     # the_model_id = 'test_bert_25000_docs_196topics_parallel'
#     # print(f"\nLoading Topic Model <{the_model_id}>...")
#     # the_topic_model = TopicModel.load(model_id=the_model_id, show_progress=True)
#     # ---------------------------------------------
#     progress_msg("Done.")
#     print(f"[{stopwatch.formatted_runtime()}]")

#     # -- Show the Topics Created --
#     total_topics = the_topic_model.num_topics
#     print(f"\n{total_topics} topics found.")
#     # ---------------------------------------------
#     print("\nTopics and Document count:")
#     all_topics = the_topic_model.top_topics()
#     for topic_and_size in all_topics:
#         print(topic_and_size)
#     # ---------------------------------------------
#     # # Topics' Vocabulary
#     # # top_n = 15
#     # # print(f"\nTop {top_n} words per topic:")
#     # # words_per_topic = the_topic_model.all_topics_top_words(top_n)
#     # # for i, word_list in words_per_topic:
#     # #     print(f"\n----> Topic <{i}>:")
#     # #     for word_sim in word_list:
#     # #         print(word_sim)

#     # # --Test Creating Hierarchically Reduced Topics--
#     # # Save the Hierarchically Reduced Topic Models.
#     # print("\nSaving Topic Model's Topic Hierarchy...")
#     # the_topic_model.save_reduced_topics(parallelism=True, show_progress=True)

#     # # -- Show Hierarchically Reduced Topics --
#     # new_topics = 10
#     # print(f"\nCreating Topic Model with {new_topics} topics.")
#     # the_topic_model.generate_new_topics(number_topics=new_topics, show_progress=True)
#     # print("Done.")
#     # print(f"[{stopwatch.formatted_runtime()}]")
#     #
#     # print("\nNew Topics and Document count:")
#     # all_topics = the_topic_model.top_topics()
#     # for topic in all_topics:
#     #     print(topic)
#     #
#     # top_n = 15
#     # print(f"\nTop {top_n} words per new topic:")
#     # words_per_topic = the_topic_model.all_topics_top_words(top_n)
#     # for i, word_list in words_per_topic:
#     #     print(f"\n----> Topic <{i}>:")
#     #     for word_sim in word_list:
#     #         print(word_sim)

#     # # --**-- Create Hierarchy for CORD-19 Topic Models --**--
#     # the_model_id = 'cord19_dataset_bert_fast'
#     # print(f"\nLoading Topic Model with ID <{the_model_id}>...")
#     # the_topic_model = TopicModel.load(model_id=the_model_id, show_progress=True)
#     # print("Done.")
#     # print(f"[{stopwatch.formatted_runtime()}]")
#     #
#     # total_topics = the_topic_model.num_topics
#     # print(f"\n{total_topics} topics found.")
#     #
#     # print("\nTopics and Document count:")
#     # all_topics = the_topic_model.top_topics()
#     # for tuple_topic_size in all_topics:
#     #     print(tuple_topic_size)
#     #
#     # # Save the Hierarchically Reduced Topic Models.
#     # print("\nSaving Topic Model's Topic Hierarchy...")
#     # the_topic_model.save_reduced_topics(show_progress=True)

#     print("\nDone.")
#     print(f"[{stopwatch.formatted_runtime()}]\n")
