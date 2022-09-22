# Gelin Eguinosa Rosique
# 2022

import multiprocessing
import numpy as np
import umap
import hdbscan
from abc import ABC, abstractmethod

from topic_corpus import TopicCorpus
from model_manager import ModelManager
from vocabulary import Vocabulary
from util_funcs import closest_vector, cosine_sim, find_top_n
from extra_funcs import progress_bar, progress_msg


# The Core Multiplier to calculate the Chunk sizes when doing Parallelism.
PARALLEL_MULT = 2
MAX_CORES = 8
PEAK_SIZE = 150


class BaseTopic(ABC):
    """
    Base Abstract Class for the Topic Models.
    """

    @property
    @abstractmethod
    def base_topic_embeds_docs(self):
        """
        Dictionary with the vector representation of the topics in the same
        vector space as the documents in the corpus.
        """
        pass

    @property
    @abstractmethod
    def base_topic_embeds_words(self):
        """
        Dictionary with the vector representation of the topics in the same
        vector space as the words in the vocabulary of the corpus. Used to
        search the words that best represent the topic.
        """
        pass

    @property
    @abstractmethod
    def base_topic_docs(self):
        """
        Dictionary with a list of the IDs of the documents belonging to each
        topic.
        """
        pass

    @property
    @abstractmethod
    def base_topic_words(self):
        """
        Dictionary with the Topic IDs as keys and the list of words that best
        describe the topics as values.
        """
        pass

    @property
    @abstractmethod
    def base_doc_embeds(self):
        """
        Dictionary with the embeddings of the documents in the corpus.
        """
        pass

    @property
    @abstractmethod
    def base_corpus_vocab(self) -> Vocabulary:
        """
        Vocabulary() class containing the words and all the data related with
        the corpus used to create the Topic Model.
        """
        pass

    @property
    def topic_ids(self):
        """
        List[str] with the IDs of the topics found by the Topic Model.
        """
        result = list(self.base_topic_embeds_docs)
        return result

    def topic_by_size(self):
        """
        Create a List of Tuples(Topic ID, Size) sorted by the amount of documents
        that each topic has.

        Returns: List[Tuples(str, int)] with the topics and their sizes.
        """
        # Get the topics and sizes.
        topic_docs = [
            (topic_id, len(doc_list))
            for topic_id, doc_list in self.base_topic_docs.items()
        ]
        # Sort the Topics by size.
        topic_docs.sort(key=lambda id_size: id_size[1], reverse=True)
        return topic_docs

    def top_words(self, topic_id: str, n=10):
        """
        Get the 'n' words that best describe the 'topic_id'.
        """
        words = self.base_topic_words[topic_id]
        if n >= len(words):
            return words
        else:
            result = words[:n]
            return result

    def topics_top_words(self, n=10):
        """
        Get the 'n' top words that best describe each of the topics in the
        model.
        """
        result_dict = {}
        for topic_id, top_words in self.base_topic_words.items():
            if n >= len(top_words):
                result_dict[topic_id] = top_words
            else:
                new_list = top_words[:n]
                result_dict[topic_id] = new_list
        # Dictionary with Top N words per topic.
        return result_dict

    def create_topic_words(self, top_n=50, min_sim=0.0, show_progress=False):
        """
        Find the 'top_n' words that best describe each topic using the words from
        the documents of the topic.

        Args:
            top_n: Int with the amount of words we want to describe the topic.
            min_sim: The minimum value of cosine similarity accepted for a word
                to describe a Topic.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        Returns:
            Dictionary(Topic ID -> List[str]) with the Topic IDs as keys and
                the list of words that best describe the topic as values.
        """
        # Progress Variables.
        count = 0
        total = len(self.topic_ids)
        # Create Set of Words per each Topic.
        topic_top_words = {}
        for topic_id in self.topic_ids:
            # Create list of words with a similarity higher than 'min_sim'.
            topic_words_sim = [
                (word, similarity)
                for word, similarity in self.topic_docs_words(topic_id)
                if similarity > min_sim
            ]
            # Get the closest 'top_n' words to the topic. (Result is Sorted).
            top_words = find_top_n(id_values=topic_words_sim, n=top_n, top_max=True)
            # Save the closest words.
            topic_top_words[topic_id] = top_words
            # Progress.
            if show_progress:
                count += 1
                progress_bar(count, total)

        # Dictionary with the Topics and their Top Words.
        return topic_top_words

    def topic_docs_words(self, topic_id: str):
        """
        Get all the words in the vocabulary of the documents belonging to the
        given 'topic_id'.

        Args:
            topic_id: String with the ID of the topic.
        Returns:
            List[Tuple(str, sim)] with the list of all the words in the
                documents and their similarity to the topic.
        """
        # Get Properties to use in the method.
        topic_docs = self.base_topic_docs
        corpus_vocab = self.base_corpus_vocab
        topic_embed = self.base_topic_embeds_words[topic_id]
        word_embeds = corpus_vocab.word_embeds

        # Create Set of Words in the Documents.
        doc_ids_sim = topic_docs[topic_id]
        topic_vocab = {
            word for doc_id, sim in doc_ids_sim
            for word in corpus_vocab.doc_words(doc_id)
        }
        # Create Tuples with the words and their similarity to the topic.
        words_sim = [
            (word, cosine_sim(topic_embed, word_embeds[word]))
            for word in topic_vocab
        ]
        # The List of Tuples of the Topic's words and their similarity.
        return words_sim


def create_docs_embeds(corpus: TopicCorpus, model: ModelManager, show_progress=False):
    """
    Get the embeddings of the documents in 'corpus' using the given Text Model
    'model', and create a dictionary with the documents IDs as keys and the
    document's embeddings as values.

    Args:
        corpus: TopicCorpus class with the documents we want to embed.
        model: ModelManager containing the Text Model used to generate the
            embeddings.
        show_progress: Bool representing whether we show the progress of the
            method or not.
    Returns:
         Dictionary (ID -> Embed) containing the IDs of the documents and their
            embeddings.
    """
    # Dictionary to store the embeddings.
    doc_embeds = {}
    # Batch Size to process documents in groups. (Speeding Up)
    batch_size = max(1, len(corpus) // 100)

    # Progress Variables.
    count = 0
    total = len(corpus)
    # Process the document (in batches to speed up process).
    processed_docs = 0
    total_docs = len(corpus.doc_ids)
    batch_count = 0
    batch_ids = []
    batch_doc_contents = []
    for doc_id in corpus.doc_ids:
        # Use title and abstract of the document to create the embedding.
        doc_content = corpus.doc_title_abstract(doc_id)
        # Add the new document ID and its content to the Batch.
        batch_count += 1
        processed_docs += 1
        batch_ids.append(doc_id)
        batch_doc_contents.append(doc_content)

        # Check if the batch is full or this is the las document of the corpus.
        if batch_count == batch_size or processed_docs == total_docs:
            # Get the embedding of the documents.
            new_embeds = model.doc_list_embeds(batch_doc_contents)
            # Add new embeddings to dictionary.
            for new_id, new_embed in zip(batch_ids, new_embeds):
                # Skip documents with nul encodings.
                if not np.any(new_embed):
                    continue
                # Save new pair ID -> Embed.
                doc_embeds[new_id] = new_embed
                # Reset batch list and counter.
                batch_count = 0
                batch_ids = []
                batch_doc_contents = []

        # Update processed documents.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Dictionary with the Documents IDs and their Embeddings.
    return doc_embeds


def find_topics(doc_embeds_list: list, show_progress=False):
    """
    Find the number or prominent topics in a Corpus given a list with the
    embeddings of the documents in the corpus.

    Args:
        doc_embeds_list: List[Numpy.ndarray] the embeddings of the documents of
            the corpus.
        show_progress: Bool representing whether we show the progress of the
            method or not.
    Returns:
        Dictionary(Topic ID -> Topic Embed) with the prominent topics found.
    """
    # UMAP to reduce the dimension of the embeddings to 5.
    if show_progress:
        progress_msg("UMAP: Reducing dimensions of the documents...")
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine')
    umap_embeds = umap_model.fit_transform(doc_embeds_list)

    # Use HDBSCAN to the find the clusters of documents in the vector space.
    if show_progress:
        progress_msg("HDBSCAN: Finding the clusters of documents in the vector space...")
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=15, metric='euclidean', cluster_selection_method='eom'
    )
    clusters_found = hdbscan_model.fit(umap_embeds)
    cluster_labels = clusters_found.labels_

    # Assign each cluster found to a new prominent topic.
    if show_progress:
        progress_msg(
            "Creating the embeddings of the prominent topics using the clusters found..."
        )
    topic_clusters_embeds = {}
    for label, doc_embed in zip(cluster_labels, doc_embeds_list):
        # Skip Noise labels.
        if str(label) == '-1':
            continue
        # Check if this is the first time we find this prominent Topic.
        topic_id = 'Topic_' + str(label)
        if topic_id in topic_clusters_embeds:
            topic_clusters_embeds[topic_id].append(doc_embed)
        else:
            topic_clusters_embeds[topic_id] = [doc_embed]

    # Progress Variables.
    count = 0
    total = len(topic_clusters_embeds)
    # Create the topic's embeddings using the average of the doc's embeddings in
    # their cluster.
    topic_embeds = {}
    for topic_id, cluster_embeds in topic_clusters_embeds.items():
        # Use Numpy to get the average embedding.
        mean_embed = np.mean(cluster_embeds, axis=0)
        topic_embeds[topic_id] = mean_embed
        # Progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Dictionary with the prominent topics and their embeds.
    return topic_embeds


def find_child_embeddings(parent_embeds: dict, child_embeds: dict,
                          parallelism=False, show_progress=False):
    """
    Given the embedding's dictionaries 'parent_embeds' and 'child_embeds',
    create a new dictionary assigning each of the child_ids to their closest
    parent_id in the embedding space.

    Args:
        parent_embeds: Dictionary containing the parent_ids as keys and their
            embeddings as values.
        child_embeds: Dictionary containing the child_ids as keys and their
            embeddings as values.
        parallelism: Bool indicating if we can use multiprocessing to speed up
            the execution of the program.
        show_progress: A Bool representing whether we show the progress of
            the method or not.
    Returns:
        Dictionary (Parent ID -> List[Child IDs]) containing the parent_ids as
            keys, and a List of the closest child_ids to them in the embedding
            space as values.
    """
    # See if we can use parallelism.
    parallel_min = 37  # Number when multicore begins to be faster that single-core.
    # parallel_min = int(2 * PEAK_SIZE / MAX_CORES)  # alternative formula (?)
    if parallelism and len(parent_embeds) > parallel_min:
        return find_children_parallel(parent_embeds, child_embeds, show_progress)

    # Check we have at least one Parent Dictionary.
    if len(parent_embeds) == 0:
        return {}

    # Progress Variables.
    count = 0
    total = len(child_embeds)
    # Iterate through each of the children and assign them to their closest parent.
    parent_child_dict = {}
    for child_id, child_embed in child_embeds.items():
        # Find the closest parent to the child.
        parent_id, similarity = closest_vector(child_embed, parent_embeds)
        # Check if we have found this parent before.
        if parent_id in parent_child_dict:
            parent_child_dict[parent_id].append((child_id, similarity))
        else:
            parent_child_dict[parent_id] = [(child_id, similarity)]
        # Show Progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Sort Children's List by their similarity to their parents.
    for tuples_child_sim in parent_child_dict.values():
        tuples_child_sim.sort(key=lambda child_sim: child_sim[1], reverse=True)
    return parent_child_dict


def find_children_parallel(parent_embeds: dict, child_embeds: dict, show_progress=False):
    """
    Version of find_child_embeddings() using parallelism.

    Given the embedding's dictionaries 'parent_embeds' and 'child_embeds',
    create a new dictionary assigning each of the child_ids to their closest
    parent_id in the embedding space.

    Args:
        parent_embeds: Dictionary containing the parent_ids as keys and their
            embeddings as values.
        child_embeds: Dictionary containing the child_ids as keys and their
            embeddings as values.
        show_progress: A Bool representing whether we show the progress of
            the method or not.
    Returns:
        Dictionary (Parent ID -> List[Child IDs]) containing the parent_ids as
            keys, and a List of the closest child_ids to them in the embedding
            space as values.
    """
    # Check we have at least one Parent Dictionary.
    if len(parent_embeds) == 0:
        return {}

    # Determine the number of cores to be used. (I made my own formula)
    optimal_cores = min(multiprocessing.cpu_count(), MAX_CORES)
    efficiency_mult = min(float(1), len(parent_embeds) / PEAK_SIZE)
    core_count = max(2, int(efficiency_mult * optimal_cores))
    # Create chunk size to process the tasks in the cores.
    chunk_size = max(1, len(child_embeds) // 100)
    # chunk_size = max(1, len(child_embeds) // (PARALLEL_MULT * core_count))
    # Create tuple parameters.
    tuple_params = [
        (child_id, child_embed, parent_embeds)
        for child_id, child_embed in child_embeds.items()
    ]

    # Create Parent-Children dictionary.
    parent_child_dict = {}
    # Create the Process Pool using the best CPU count (formula) for the task.
    with multiprocessing.Pool(processes=core_count) as pool:
        # Use Pool.imap() if we have to show the method's progress.
        if show_progress:
            # Report Parallelization.
            progress_msg(f"Using Parallelization <{core_count} cores>")
            # Progress Variables.
            count = 0
            total = len(child_embeds)
            # Create Lazy iterator of results using Pool.imap().
            imap_results_iter = pool.imap(
                _custom_closest_vector, tuple_params, chunksize=chunk_size
            )
            # Add each child to its closest parent.
            for child_id, parent_id, similarity in imap_results_iter:
                # Check if we have found this 'parent_id' before.
                if parent_id in parent_child_dict:
                    parent_child_dict[parent_id].append((child_id, similarity))
                else:
                    parent_child_dict[parent_id] = [(child_id, similarity)]
                # Progress.
                count += 1
                progress_bar(count, total)
        # No need to report progress, use Pool.map()
        else:
            # Find the closest parent to each child.
            tuple_results = pool.map(
                _custom_closest_vector, tuple_params, chunksize=chunk_size
            )
            # Save the closest children to each Parent.
            for child_id, parent_id, similarity in tuple_results:
                # Check if we have found this 'parent_id' before.
                if parent_id in parent_child_dict:
                    parent_child_dict[parent_id].append((child_id, similarity))
                else:
                    parent_child_dict[parent_id] = [(child_id, similarity)]

    # Sort Children's List by their similarity to their parents.
    for tuples_child_sim in parent_child_dict.values():
        tuples_child_sim.sort(key=lambda child_sim: child_sim[1], reverse=True)
    return parent_child_dict


def _custom_closest_vector(id_embed_parents: tuple):
    """
    Custom-made version of the method closest_vector to use in the methods
    find_children_parallel() and children_count_parallel().

    Take the 'child_id' and 'child_embed', with the parent_embeds inside the
    'id_dicts_tuple' parameter, and find the closest parent embedding to the
    child embedding with the provided 'child_id'.

    Args:
        id_embed_parents: Tuple ('child_id', 'child_embed', 'parent_embeds')
            containing the 'child_embed' and 'parent_embeds' to call the
            function closest_vector().
    Returns:
        Tuple with the 'child_id', its closest 'parent_id' and their 'similarity'.
    """
    child_id, child_embed, parent_embeds = id_embed_parents
    parent_id, similarity = closest_vector(child_embed, parent_embeds)
    return child_id, parent_id, similarity


def best_midway_sizes(original_size: int):
    """
    Create a set containing the topic sizes for whom we are going to save a
    Hierarchically Reduced Topic Model to speed up the process of returning
    a Topic Model when a user requests a custom size.

    The Topic Sizes will have a different step depending on the range of topic
    sizes:
    - Base topic size of 2.
    - Step of  5 between  5 and 30.
    - Step of 10 between 30 and 100.
    - Step of 25 between 100 and 300.
    - Step of 50 between 300 and 1000.

    Example: 2, 5, 10, ..., 25, 30, 40,..., 90, 100, 125, ..., 275, 300, 350...

    Args:
        original_size: Int with the size of the original Topic Model.

    Returns:
        Set containing the Ints with the topic sizes we have to save when
            generating the reduced Topic Model.
    """
    # Check we don't have an Original Topic of size 2.
    midway_sizes = set()
    if original_size > 2:
        midway_sizes.add(2)
    # Sizes between 5 and 30.
    midway_sizes.update(range(5, min(30, original_size), 5))
    # Sizes between 30 and 100.
    midway_sizes.update(range(30, min(100, original_size), 10))
    # Sizes between 100 and 300.
    midway_sizes.update(range(100, min(300, original_size), 25))
    # Sizes between 300 and 1000.
    midway_sizes.update(range(300, min(1_001, original_size), 50))

    # The Intermediate sizes to create a reference Hierarchical Topic Model.
    return midway_sizes
