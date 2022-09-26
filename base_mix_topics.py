# Gelin Eguinosa Rosique
# 2022
import json

import numpy as np
import umap
import hdbscan
from os import mkdir
from shutil import rmtree
from os.path import isdir, join
from abc import ABC, abstractmethod

from base_topics import (
    BaseTopics, refresh_topic_ids, closest_vector, count_child_embeddings
)
from topic_corpus import TopicCorpus
from model_manager import ModelManager
from util_funcs import dict_ndarray2list, dict_list2ndarray
from extra_funcs import progress_bar, progress_msg  # , big_number


class BaseMixTopics(BaseTopics, ABC):
    """
    Base Abstract Class for the Mixed Topic Models with two vector spaces, one
    for the Specter embeddings of the documents, and another for the embeddings
    of the words in the vocabulary.
    """

    # --------------------------------------------
    # Abstract Properties
    # --------------------------------------------

    @property
    @abstractmethod
    def base_topic_embeds_docs(self) -> dict:
        """
        Dictionary with the vector representation of the topics in the same
        vector space as the documents in the corpus.
        """
        pass

    @property
    @abstractmethod
    def base_topic_embeds_words(self) -> dict:
        """
        Dictionary with the vector representation of the topics in the same
        vector space as the words in the vocabulary of the corpus.
        """
        pass

    @property
    @abstractmethod
    def base_red_topic_embeds_docs(self) -> dict:
        """
        Dictionary with the vector representation of the Reduced Topics in the
        same vector space as the documents in the corpus.
        """
        pass

    # --------------------------------------------
    # BaseTopics Properties
    # --------------------------------------------

    @property
    def base_topic_embeds(self) -> dict:
        """
        Dictionary with the vector representation of the topics in the same
        vector space as the documents in the corpus.
        """
        return self.base_topic_embeds_docs

    @property
    def base_cur_topic_embeds(self) -> dict:
        """
        Dictionary with the vector representation of the Reduced Topics in the
        same vector space as the documents in the corpus.
        """
        return self.base_red_topic_embeds_docs

    def base_reduce_topics(self, new_size: int, parallelism=False, show_progress=False):
        """
        Create a new Hierarchical Topic Model with specified number of topics
        (new_size). The 'new_size' needs to be at least 2, and smaller than the
        current number of topics in the model.

        Args:
            new_size: Int with the desired topic count for the Model.
            parallelism: Bool indicating if we can use multiprocessing or not.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        Returns:
            Dictionaries with the Topic Embeddings of the new Reduced Topic
            Model, both in the Document Space and Vocabulary Space.
        """
        # Check the topic size requested is valid.
        if not 1 < new_size < self.topic_size:
            # Invalid topic size requested.
            raise Exception(
                    f"Invalid reduced topic size of {new_size} requested, when "
                    f"the Topic Model only has {self.topic_size} topics."
                )
        # Check if we have saved Reduced Topic Models.
        if not self.reduced_topics_saved():
            # No - Start Reducing from the Total Size of Topics in the Corpus.
            current_size = self.topic_size
            new_topic_embeds_docs = self.base_topic_embeds_docs.copy()
            new_topic_embeds_words = self.base_topic_embeds_words.copy()
            new_topic_sizes = dict(
                [(topic_id, len(doc_list))
                 for topic_id, doc_list in self.base_topic_docs.items()]
            )
        else:
            # Yes - Get the closest Reduced Topics.
            main_sizes = self.main_sizes(self.topic_size)
            closest_size = min(size for size in main_sizes if size >= new_size)
            # Upload the Reduced Topic Model.
            if show_progress:
                progress_msg(f"Loading Reduced model with {closest_size} topics...")
            reduced_topic_file = self.base_reduced_prefix + str(closest_size) + '.json'
            model_folder_path = join(self.base_class_folder, self.model_folder_name)
            reduced_folder_path = join(model_folder_path, self.base_reduced_folder)
            reduced_topic_path = join(reduced_folder_path, reduced_topic_file)
            with open(reduced_topic_path, 'r') as f:
                reduced_topic_index = json.load(f)
            # Get the Reduced Topic Sizes.
            new_topic_sizes = reduced_topic_index['topic_sizes']
            json_topic_embeds_docs = reduced_topic_index['topic_embeds_docs']
            json_topic_embeds_words = reduced_topic_index['topic_embeds_words']
            current_size = len(new_topic_sizes)
            new_topic_embeds_docs = dict_list2ndarray(json_topic_embeds_docs)
            new_topic_embeds_words = dict_list2ndarray(json_topic_embeds_words)

        # Reduce Topic Size until we get the desired number of topics.
        if current_size > new_size and show_progress:
            progress_msg(
                f"Reducing Topic Model from {current_size} topics to {new_size} "
                f"topics..."
            )
        while current_size > new_size:
            # Reduce the Topic Size by 1.
            if show_progress:
                progress_msg(
                    f"Reducing from {current_size} to {current_size - 1} topics..."
                )
            result_tuple = self.reduce_mix_topic_size(
                ref_topic_embeds_docs=new_topic_embeds_docs,
                ref_topic_embeds_words=new_topic_embeds_words,
                topic_sizes=new_topic_sizes, parallelism=parallelism,
                show_progress=show_progress
            )
            new_topic_embeds_docs = result_tuple[0]
            new_topic_embeds_words = result_tuple[1]
            new_topic_sizes = result_tuple[2]
            # Update the current number of topics.
            current_size = len(new_topic_sizes)

        # Ready - Dictionaries with the Topic Embeddings of the Reduced Model.
        if show_progress:
            progress_msg(f"Topic Model reduced to {current_size} topics.")
        return new_topic_embeds_docs, new_topic_embeds_words

    def base_save_reduced_topics(
            self, parallelism=False, override=False, show_progress=False
    ):
        """
        Create a list of basic topic sizes between 2 and the size of the current
        Topic Model, to create and save the Hierarchical Topic Models of this
        Model with these sizes, so when we create a new Hierarchical Topic Model
        we can do it faster, only having to start reducing the Topic sizes from
        the closest basic topic size.

        The saved topic sizes will be in the range of 2-1000, with different
        steps depending on the Topic Size range.
          - Step of  5 between  2 and 30.
          - Step of 10 between 30 and 100.
          - Step of 25 between 100 and 300.
          - Step of 50 between 300 and 1000.

        Args:
            parallelism: Bool to indicate if we can use multiprocessing to speed
                up the runtime of the method.
            override: Bool indicating if we can delete a previously saved
                Reduced Topics.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check the Model is Saved.
        if show_progress:
            progress_msg("Checking the given ID corresponds to the current Topic Model...")
        if not self.model_saved(model_id=self.base_model_id):
            raise Exception(
                "The Model needs to be saved before saving its Reduced Topics."
            )
        # Check we can create a Reduced Topic Model.
        if self.topic_size <= 2:
            return
        # Check if the Model has Reduced Topics Saved.
        if not override and self.reduced_topics_saved():
            raise FileExistsError(
                "This Topic Model already has its Reduced Topics saved. Set "
                "the 'override' parameter to 'True' to replace them with new "
                "Reduced Topics. "
            )

        # Create Reduced Topics Folder.
        model_folder_path = join(self.base_class_folder, self.model_folder_name)
        reduced_folder_path = join(model_folder_path, self.base_reduced_folder)
        if isdir(reduced_folder_path):
            # Remove the previously saved Hierarchy.
            rmtree(reduced_folder_path)
        mkdir(reduced_folder_path)

        # Get the Set of Reduced Topic Sizes we have to save.
        main_sizes = self.main_sizes(self.topic_size)
        # Initialize the Topic Reduction Dictionaries.
        current_size = self.topic_size
        new_topic_embeds_docs = self.base_topic_embeds_docs.copy()
        new_topic_embeds_words = self.base_topic_embeds_words.copy()
        new_topic_sizes = dict(
            [(topic_id, len(doc_list))
             for topic_id, doc_list in self.base_topic_docs.items()]
        )
        # Start Reducing Topics and Saving the Main Topic Sizes.
        if show_progress:
            progress_msg("Saving the Hierarchically Reduced Topic Models...")
        while current_size > 2:
            # Reduce Topic Size by 1.
            if show_progress:
                progress_msg(
                    f"Reducing from {current_size} to {current_size -1} topics..."
                )
            result_tuple = self.reduce_mix_topic_size(
                ref_topic_embeds_docs=new_topic_embeds_docs,
                ref_topic_embeds_words=new_topic_embeds_words,
                topic_sizes=new_topic_sizes, parallelism=parallelism,
                show_progress=show_progress
            )
            new_topic_embeds_docs = result_tuple[0]
            new_topic_embeds_words = result_tuple[1]
            new_topic_sizes = result_tuple[2]
            # Update the current number of topics.
            current_size = len(new_topic_sizes)
            # Check if we have to save the current embeddings and sizes.
            if current_size in main_sizes:
                if show_progress:
                    progress_msg(
                        "<<Main Topic Found>>\n"
                        f"Saving Reduced Topic Model with {current_size} topics..."
                    )
                # Transform Topic Embeddings to List[float].
                json_topic_embeds_docs = dict_ndarray2list(new_topic_embeds_docs)
                json_topic_embeds_words = dict_ndarray2list(new_topic_embeds_words)
                # Create Dict with embeddings and sizes.
                reduced_topics_index = {
                    'topic_embeds_docs': json_topic_embeds_docs,
                    'topic_embeds_words': json_topic_embeds_words,
                    'topic_sizes': new_topic_sizes
                }
                # Save Index of the current Reduced Topics.
                reduced_topics_file = (
                    self.base_reduced_prefix + str(current_size) + '.json'
                )
                reduced_topics_path = join(reduced_folder_path, reduced_topics_file)
                with open(reduced_topics_path, 'w') as f:
                    json.dump(reduced_topics_index, f)
                if show_progress:
                    progress_msg("<<Saved>>")

    def reduce_mix_topic_size(
            self, ref_topic_embeds_docs: dict, ref_topic_embeds_words: dict,
            topic_sizes: dict, parallelism=False, show_progress=False
    ):
        """
        Reduce by 1 the number of topics inside the dictionaries
        'ref_topics_embeds_docs' and 'ref_topics_embeds_words', joining the
        smallest topic to its closest neighbor.

        The embed dictionaries are treated as a reference and will be modified
        to store the new reduced topics.

        Args:
            ref_topic_embeds_docs: Dictionary containing the embeddings of the
                topics in the Document space.
            ref_topic_embeds_words: Dictionary containing the embeddings of the
                topics in the Vocabulary space.
            topic_sizes: Dictionary containing the current size of the topics we
                are reducing.
            parallelism: Bool to indicate if we have to use the multiprocessing
                version of this function.
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns:
            Tuple with the dictionaries 'ref_topic_embeds_docs' and
                'ref_topic_embeds_words' with the embeddings of the new topics,
                and the dictionary 'topic_sizes' containing the updated sizes
                for the new Topics.
        """
        # Get the ID and Embeds of the Smallest Topic.
        min_topic_id = min(
            topic_sizes.keys(), key=lambda topic_id: topic_sizes[topic_id]
        )
        min_embed_docs = ref_topic_embeds_docs[min_topic_id]
        min_embed_words = ref_topic_embeds_words[min_topic_id]

        # Delete Smallest Topic.
        del ref_topic_embeds_docs[min_topic_id]
        del ref_topic_embeds_words[min_topic_id]
        # Get the closest topic to the Smallest Topic.
        close_topic_id, _ = closest_vector(min_embed_docs, ref_topic_embeds_docs)
        close_embed_docs = ref_topic_embeds_docs[close_topic_id]
        close_embed_words = ref_topic_embeds_words[close_topic_id]
        # Merge the embeddings of the topics.
        min_size = topic_sizes[min_topic_id]
        close_size = topic_sizes[close_topic_id]
        total_size = min_size + close_size
        new_topic_embed_docs = (
            (min_size * min_embed_docs + close_size * close_embed_docs) / total_size
        )
        new_topic_embed_words = (
            (min_size * min_embed_words + close_size * close_embed_words) / total_size
        )
        # Update Embeddings of the closest topic.
        ref_topic_embeds_docs[close_topic_id] = new_topic_embed_docs
        ref_topic_embeds_words[close_topic_id] = new_topic_embed_words

        # Update Topic Sizes.
        new_size = len(ref_topic_embeds_docs)
        if show_progress:
            progress_msg(f"Creating sizes for the new {new_size} topics...")
        new_topic_sizes = count_child_embeddings(
            parent_embeds=ref_topic_embeds_docs, child_embeds=self.base_doc_embeds,
            parallelism=parallelism, show_progress=show_progress
        )
        # Dictionaries with new embeds and topic sizes.
        return ref_topic_embeds_docs, ref_topic_embeds_words, new_topic_sizes


def create_specter_embeds(corpus: TopicCorpus, load_full_dict=False, show_progress=False):
    """
    Create dictionary with the Specter embeddings of the documents in the
    'corpus' using the embeddings created by the Cord-19 dataset.

    Args:
        corpus: CorporaManager class with the documents we want to embed.
        load_full_dict: Bool indicating if we can load the dictionary with all
            the specter embeddings of the documents.
        show_progress: Bool representing whether we show the progress of the
            method or not.
    Returns:
        Dictionary (Doc_ID -> Numpy.ndarray) containing the IDs of the documents
            and their Specter Embeddings.
    """
    # Check if we can load the full Dict with the Specter Embeddings of the Corpus.
    if load_full_dict:
        if show_progress:
            progress_msg("Loading Specter Embeddings Dictionary...")
        corpus.load_embeddings_dict()

    # Create Dictionary with the Docs and their embeddings.
    count = 0
    total = len(corpus)
    doc_embeds = {}
    for doc_id in corpus.doc_ids:
        # Get the Embed of the document.
        list_embed = corpus.doc_specter_embed(doc_id)
        doc_embed = np.array(list_embed)
        doc_embeds[doc_id] = doc_embed
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Unload Full Dictionary if it was loaded before.
    if load_full_dict:
        if show_progress:
            progress_msg("Unloading Specter Embeddings Dictionary...")
        corpus.unload_embeddings_dict()

    # Dictionary with the Docs and their specter embeddings.
    return doc_embeds


def find_mix_topics(doc_embeds: dict, corpus: TopicCorpus, text_model: ModelManager,
                    show_progress=False):
    """
    Find the number of prominent topics in the corpus, given the Specter embeddings
    of its documents, and create the vector representation of the topics in the
    Document Space (Specter) and in the Vocabulary Space (text_model).

    Args:
        doc_embeds: Dictionary with the specter embeddings of the documents.
        corpus: CorporaManager with the content of the documents we are
            processing.
        text_model: TopicCorpus used to create the vector representation of the
            topics in the Vocabulary Space.
        show_progress: Bool representing whether we show the progress of the
            method or not.
    Returns:
        Two Dictionaries with the embeddings of the Topics in the Document Space
            and the Vocabulary Space.
    """
    # Create Document Embeddings with 5 dimensions.
    doc_ids = list(doc_embeds.keys())
    embeds_list = list(doc_embeds.values())
    if show_progress:
        progress_msg("UMAP: Reducing dimensions of the documents...")
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine')
    umap_embeds = umap_model.fit_transform(embeds_list)

    # Use HDBSCAN to find the clusters of documents in the Specter vector space.
    if show_progress:
        progress_msg("HDBSCAN: Finding the clusters of documents in the vector space...")
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=15, metric='euclidean', cluster_selection_method='eom'
    )
    clusters_found = hdbscan_model.fit(umap_embeds)
    cluster_labels = clusters_found.labels_

    # Assign each cluster found to a new prominent topic, save their doc_ids.
    if show_progress:
        progress_msg("Assign each document in a cluster to a topic...")
    topic_clusters_docs = {}
    for label, doc_id in zip(cluster_labels, doc_ids):
        # Skip Noise Labels.
        if label == -1:
            continue
        # Check if this is the first time we found this prominent Topic.
        topic_id = 'Topic_' + str(label)
        if topic_id in topic_clusters_docs:
            topic_clusters_docs[topic_id].append(doc_id)
        else:
            topic_clusters_docs[topic_id] = [doc_id]
    # Report number of topics.
    topic_size = len(topic_clusters_docs)
    if show_progress:
        progress_msg(f"{topic_size} topics found.")

    # Refresh the Topic IDs (So they are correctly formatted).
    topic_clusters_docs = refresh_topic_ids(topic_dict=topic_clusters_docs)
    # # Report Prominent Topics cluster sizes.
    # if show_progress:
    #     progress_msg("<< Prominent Topics cluster sizes >>")
    #     for topic_id, cluster_doc_ids in topic_clusters_docs.items():
    #         progress_msg(f" -> {topic_id}: {big_number(len(cluster_doc_ids))} docs")

    # Progress Variables.
    count = 0
    total = len(topic_clusters_docs)
    # Create the Topic Embeddings in the Document & Vocabulary Space using the
    # documents in their cluster.
    topic_embeds_docs = {}
    topic_embeds_words = {}
    if show_progress:
        progress_msg(
            f"Creating the Doc & Vocab Embeds of the {topic_size} topics found..."
        )
    for topic_id, cluster_doc_ids in topic_clusters_docs.items():
        # Create Topic Embed in the Document Space.
        specter_cluster_embeds = [doc_embeds[doc_id] for doc_id in cluster_doc_ids]
        specter_mean_embed = np.mean(specter_cluster_embeds, axis=0)
        topic_embeds_docs[topic_id] = specter_mean_embed

        # Create Topic Embed in the Vocabulary Space.
        cluster_docs_content = [
            corpus.doc_title_abstract(doc_id)
            for doc_id in cluster_doc_ids
        ]
        vocab_cluster_embeds = text_model.doc_list_embeds(cluster_docs_content)
        vocab_mean_embed = np.mean(vocab_cluster_embeds, axis=0)
        topic_embeds_words[topic_id] = vocab_mean_embed

        # Progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Dictionaries with the embeddings of the Prominent Topics.
    return topic_embeds_docs, topic_embeds_words
