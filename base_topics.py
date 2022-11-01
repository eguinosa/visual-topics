# Gelin Eguinosa Rosique
# 2022

import json
import multiprocessing
import numpy as np
import umap
import hdbscan
from os import mkdir
from shutil import rmtree
from os.path import isdir, isfile, join
from abc import ABC, abstractmethod

from topic_corpus import TopicCorpus
from model_manager import ModelManager
from vocabulary import Vocabulary
from extra_funcs import progress_bar, progress_msg, number_to_size
from util_funcs import (
    closest_vector, cosine_sim, find_top_n, dict_list2ndarray,
    dict_ndarray2list, mean_similarity
)


# The Core Multiplier to calculate the Chunk sizes when doing Parallelism.
PEAK_ITERATIONS = 3_750_000  # 150 topics * 25,000 docs
BASE_ITERATIONS = 925_000  # 37 topics * 25,000 docs
MAX_CORES = 8


class BaseTopics(ABC):
    """
    Base Abstract Class for the Topic Models.
    """

    @property
    @abstractmethod
    def base_model_id(self) -> str:
        """
        String the ID used to identify the current Topic Model.
        """
        pass

    @property
    @abstractmethod
    def base_corpus_id(self) -> str:
        """
        String with the ID of the corpus used to create the Topic Model.
        """
        pass

    @property
    @abstractmethod
    def base_text_model_name(self) -> str:
        """
        String with the name of the Text Model used to create the embeddings
        of the topics, documents and words in the same vector space.
        """
        pass

    @property
    @abstractmethod
    def base_topic_embeds(self) -> dict:
        """
        Dictionary with the vector representation of the topics in the same
        vector space as the documents in the corpus.
        """
        pass

    @property
    @abstractmethod
    def base_cur_topic_embeds(self) -> dict:
        """
        Dictionary with the vector representation of the current Topics in the
        same vector space as the documents in the corpus.

        When the Base Current Topic Embeds do not return None, then the model
        assumes that the Reduced Topics of the Model are ready and available.
        """
        pass

    @property
    @abstractmethod
    def base_topic_docs(self) -> dict:
        """
        Dictionary with a list of the IDs of the documents belonging to each
        topic.
        """
        pass

    @property
    @abstractmethod
    def base_cur_topic_docs(self) -> dict:
        """
        Dictionary with the list of Documents (IDs) that belong to each of the
        current Topics.
        """
        pass

    @property
    @abstractmethod
    def base_topic_words(self) -> dict:
        """
        Dictionary with the Topic IDs as keys and the list of words that best
        describe the topics as values.
        """
        pass

    @property
    @abstractmethod
    def base_topics_homogeneity(self) -> dict:
        """
        Dictionary with the Homogeneity of the Topics in the Model.
        """
        pass

    @base_topics_homogeneity.setter
    @abstractmethod
    def base_topics_homogeneity(self, value):
        pass

    # @base_topics_homogeneity.deleter
    # @abstractmethod
    # def base_topics_homogeneity(self):
    #     pass

    @property
    @abstractmethod
    def base_cur_topic_words(self):
        """
        Dictionary the list of words that best describe each of the current
        Topics.
        """
        pass

    @property
    @abstractmethod
    def base_doc_embeds(self) -> dict:
        """
        Dictionary with the embeddings of the documents in the corpus.
        """
        pass

    @property
    @abstractmethod
    def base_corpus_vocab(self) -> Vocabulary:
        """
        Vocabulary class created with the corpus of the Topic Model.
        """
        pass

    @property
    @abstractmethod
    def base_class_folder(self) -> str:
        """
        String with the name of the folder where the models of the class will
        be stored.
        """
        pass

    @property
    @abstractmethod
    def model_folder_name(self) -> str:
        """
        Name of the folder where the Topic Model will be stored.
        """
        pass

    @property
    @abstractmethod
    def base_reduced_folder(self) -> str:
        """
        String with the name of the folder where the Reduced Topic Models will
        be stored.
        """
        pass

    @property
    @abstractmethod
    def base_reduced_prefix(self) -> str:
        """
        String with the prefix used to create the name of the files used store
        the reduced Topics.
        """
        pass

    @abstractmethod
    def reduce_topics(
            self, new_size: int, parallelism: bool, show_progress: bool
    ) -> None:
        """
        Create a new Hierarchical Topic Model with specified number of topics
        (new_size). The 'new_size' needs to be at least 2, and smaller than the
        current number of topics in the model.
        """
        pass

    @abstractmethod
    def refresh_vocabulary(self, show_progress: bool) -> None:
        """
        Remake the Corpus Vocabulary of the Topic Model using the saved info
        about the Corpus and Text Model used.
        """
        pass

    @abstractmethod
    def save(self, show_progress: bool) -> None:
        """
        Save the Main Topic Model's Attributes, so the model can be loaded later
        using the given or created model ID.
        """
        pass

    @abstractmethod
    def save_reduced_topics(
            self, parallelism: bool, override: bool, show_progress: bool
    ) -> None:
        """
        Use the base_save_reduced_topics() method to save the Hierarchically
        Reduced Topic Models for the main sizes, and update the information in
        the Basic Index to show that the model now has Reduced Topics.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_id: str, show_progress: bool):
        """
        Load a saved Topic Model using its 'model_id'.
        """
        pass

    @classmethod
    @abstractmethod
    def basic_info(cls, model_id: str) -> dict:
        """
        Load the Basic Info of the Topic Model 'model_id'. Basic Information
        attributes:

        -  topic_size: Int
        -  corpus_size: Int
        -  corpus_id: String
        -  text_model_name: String
        -  has_reduced_topics: Bool
        """
        pass

    @classmethod
    @abstractmethod
    def model_saved(cls, model_id: str) -> bool:
        """
        Check if the given 'model_id' string corresponds to a saved Topic Model.
        """
        pass

    @classmethod
    @abstractmethod
    def saved_models(cls) -> list:
        """
        Create a list the IDs of the saved Topic Models.
        """
        pass

    @property
    def current_doc_space_topic_embeds(self):
        """
        The embeddings of the topics in the same vector space as the documents
        of the corpus. Returns the current (reduced) topics if possible,
        otherwise returns the original topics.
        """
        # Check if we have a reduced topic size.
        if self.base_cur_topic_embeds:
            return self.base_cur_topic_embeds
        else:
            return self.base_topic_embeds

    @property
    def current_word_space_topic_embeds(self):
        """
        The embeddings of the topics in the same vector space as the words in
        the vocabulary of the corpus. Returns the current (reduced) topics if
        possible, otherwise returns the original topics.
        """
        # Check if we have reduced topics.
        if self.base_cur_topic_embeds:
            return self.base_cur_topic_embeds
        else:
            return self.base_topic_embeds

    @property
    def doc_space_doc_embeds(self):
        """
        Get the embeddings of the documents in the vector space where the topics
        and the documents are.
        """
        return self.base_doc_embeds

    @property
    def word_space_doc_embeds(self):
        """
        Get the embeddings of the documents in the vector space where the
        topics, documents and words are all together.
        """
        return self.base_doc_embeds

    @property
    def topic_size(self):
        """
        Int with the number of Topic the model has.
        """
        return len(self.base_topic_embeds)

    @property
    def cur_topic_size(self):
        """
        Int with the Number of Topics the Current Model has.
        """
        if self.base_cur_topic_embeds:
            return len(self.base_cur_topic_embeds)
        else:
            return self.topic_size

    @property
    def topic_ids(self):
        """
        List[str] with the IDs of the topics found by the Topic Model.
        """
        id_list = list(self.base_topic_embeds.keys())
        return id_list

    @property
    def cur_topic_ids(self):
        """
        List[str] with the IDs of the current topics in the Model.
        """
        if self.base_cur_topic_embeds:
            id_list = list(self.base_cur_topic_embeds.keys())
            return id_list
        else:
            return self.topic_ids

    @property
    def has_reduced_topics(self):
        """
        Bool indicating if the Model has Reduced Topics.
        """
        if not self.base_cur_topic_embeds:
            return False
        # Check if the Current Embeds has fewer Topics than the Original.
        if len(self.base_cur_topic_embeds) < len(self.base_topic_embeds):
            return True
        else:
            return False

    def topic_by_size(self):
        """
        Create a List of Tuples(Topic ID, Size) sorted by the amount of documents
        that each topic has.

        Returns: List[Tuples(str, int)] with the topics and their sizes.
        """
        # Get the topics and sizes.
        topic_size = [
            (topic_id, len(doc_list))
            for topic_id, doc_list in self.base_topic_docs.items()
        ]
        # Sort the Topics by size.
        topic_size.sort(key=lambda id_size: id_size[1], reverse=True)
        return topic_size

    def cur_topic_by_size(self):
        """
        Create List of Tuples with the ID and Size for each of the current
        topics.

        Returns: List[Tuples(str, int)] with the current topics and their sizes.
        """
        # Get Current Topics and Sizes.
        cur_topic_sizes = [
            (cur_topic_id, len(doc_list))
            for cur_topic_id, doc_list in self.base_cur_topic_docs.items()
        ]
        # Sort the Current Topics by size.
        cur_topic_sizes.sort(key=lambda id_size: id_size[1], reverse=True)
        return cur_topic_sizes

    def cur_topic_by_homogeneity(self, show_progress=False):
        """
        Create List of Tuples with the ID and Homogeneity of each of the current
        topics.

        Returns: List[Tuples(str, float)] with the current topics and their
            homogeneity.
        """
        if show_progress:
            cur_topics_homogeneity = []
            for cur_topic_id in self.cur_topic_ids:
                progress_msg(f"Calculating Homogeneity of <{cur_topic_id}>...")
                homogeneity = self.cur_topic_homogeneity(
                    cur_topic_id=cur_topic_id, parallelism=True,
                    show_progress=show_progress
                )
                cur_topics_homogeneity.append((cur_topic_id, homogeneity))
        else:
            cur_topics_homogeneity = [
                (cur_topic_id,
                 self.cur_topic_homogeneity(
                     cur_topic_id=cur_topic_id, parallelism=True,
                     show_progress=show_progress
                 ))
                for cur_topic_id in self.cur_topic_ids
            ]
        # Topics & their Homogeneity.
        cur_topics_homogeneity.sort(key=lambda id_homo: id_homo[1], reverse=True)
        return cur_topics_homogeneity

    def cur_topic_by_pwi(self, word_num=-1, pwi_type='exact'):
        """
        Create List of Tuples with the ID and PWI value for each of the current
        topics.
        """
        # Get the Topics and Their PWI.
        cur_topics_pwi = [
            (cur_topic_id,
             self.cur_topic_pwi(
                 cur_topic_id=cur_topic_id, word_num=word_num, pwi_type=pwi_type
             ))
            for cur_topic_id in self.base_cur_topic_embeds.keys()
        ]
        # Sort the Current Topics by their PWI.
        cur_topics_pwi.sort(key=lambda id_pwi: id_pwi[1], reverse=True)
        return cur_topics_pwi

    def top_words_topic(self, topic_id: str, top_n=10, comparer='cos-sim'):
        """
        Get the 'n' words that best describe the 'topic_id'. The Top Words can
        be retrieved using their similarity to the Topic Embedding, or their
        descriptive value, determined using the PWI formula.

        The 'comparer' determines which formula will be used to determine the
        top words in the topic, it can be 'cos-sim' for cosine similarity,
        'pwi-exact' for Mutual Information with the exact formula, or
        'pwi-tf-idf' for the Mutual Information with the tf-idf formula.

        Args:
            topic_id: String with the ID of the Document.
            top_n: Int with the number of words we want to get from the
                topic.
            comparer: The formula used to compare the words that best describe
                the topic 'cos-sim', 'pwi-exact' or 'pwi-tf-idf'.
        Returns:
            List[Tuple(word, value)] with the list of tuples of the top words in
                the topic with the value used to compare the word to the others.
        """
        # Using PWI.
        if comparer in {'pwi-exact', 'pwi-tf-idf'}:
            pwi_type = comparer[4:]
            words_pwi = self.topic_words_pwi(topic_id, pwi_type)
            # Get the Top Words with the Highest PWI Value.
            top_words = {
                word
                for word, pwi_value in find_top_n(words_pwi.items(), n=top_n)
            }
            # Still sort words by their similarity to the Topic.
            words_tuple = [
                (word, sim)
                for word, sim in self.base_topic_words[topic_id]
                if word in top_words
            ]
            return words_tuple
        if comparer != 'cos-sim':
            raise NameError(
                f"The Comparer <{comparer}> is not supported to sort the words "
                f"in the topic."
            )
        # Using the default 'cos-sim' a.k.a. cosine similarity.
        words_sims = self.base_topic_words[topic_id]
        if top_n >= len(words_sims):
            return words_sims
        else:
            result_tuples = words_sims[:top_n]
            return result_tuples

    def top_words_cur_topic(self, cur_topic_id: str, top_n=10, comparer='cos-sim'):
        """
        Get the 'n' words that best describe the 'cur_topic_id'. The Top Words
        can be retrieved using their similarity to the Topic Embedding, or their
        descriptive value, determined using the PWI formula.

        The 'comparer' determines which formula will be used to determine the
        top words in the topic, it can be 'cos-sim' for cosine similarity,
        'pwi-exact' for Mutual Information with the exact formula, or
        'pwi-tf-idf' for the Mutual Information with the tf-idf formula.

        Args:
            cur_topic_id: String with the ID of the Document.
            top_n: Int with the number of words we want to get from the
                topic.
            comparer: The formula used to compare the words that best describe
                the topic 'cos-sim', 'pwi-exact' or 'pwi-tf-idf'.
        Returns:
            List[Tuple(word, value)] with the list of tuples of the top words in
                the topic with the value used to compare the word to the others.
        """
        # Using PWI.
        if comparer in {'pwi-exact', 'pwi-tf-idf'}:
            pwi_type = comparer[4:]
            words_pwi = self.cur_topic_words_pwi(cur_topic_id, pwi_type)
            # Get the Top Words with Higher PWI value.
            top_words = {
                word
                for word, pwi_value in find_top_n(words_pwi.items(), n=top_n)
            }
            # Still sort words by their similarity to the Topic.
            words_tuple = [
                (word, sim)
                for word, sim in self.base_cur_topic_words[cur_topic_id]
                if word in top_words
            ]
            return words_tuple
        if comparer != 'cos-sim':
            raise NameError(
                f"The Comparer <{comparer}> is not supported to sort the words "
                f"in the current topic."
            )
        # Using the default 'cos-sim' a.k.a. cosine similarity.
        words_sims = self.base_cur_topic_words[cur_topic_id]
        if top_n >= len(words_sims):
            return words_sims
        else:
            result_tuples = words_sims[:top_n]
            return result_tuples

    def topics_top_words(self, top_n=10, comparer='cos-sim'):
        """
        Get the 'n' top words that best describe each of the topics in the
        model.
        """
        result_dict = {}
        for topic_id, _ in self.topic_by_size():
            result_dict[topic_id] = self.top_words_topic(
                topic_id=topic_id, top_n=top_n, comparer=comparer
            )
        # Dictionary with Top N words per topic.
        return result_dict

    def cur_topic_varied_words(self, cur_topic_id: str, top_n=10, sample_ratio=1.6):
        """
        Create a diversified description of the current topic 'cur_topic_id'
        with 'top_n' words, taking the words from the closest 'sample_size'
        words to the topic in the vector space.
        """
        # Check we have a valid sample ration, above 1.2.
        if sample_ratio < 1.2:
            raise Exception(
                "The Value of sample_ration needs to be higher than 1.2."
            )
        # Create the sample size.
        sample_size = round(top_n * sample_ratio)
        # Get the Embeddings of the Word in the Vocabulary.
        word_embeds = self.base_corpus_vocab.word_embeds

        # Get the words used to create the Varied Description.
        topic_words_sims = self.base_cur_topic_words[cur_topic_id]
        # Get the Top Word Info.
        top_word_tuple = topic_words_sims[0]
        top_word = top_word_tuple[0]
        top_word_embed = word_embeds[top_word]
        second_word_tuple = topic_words_sims[1]
        second_word = second_word_tuple[0]
        second_word_embed = word_embeds[second_word]

        # Add the most diverse word in every iteration.
        sample_words = topic_words_sims[2:sample_size]
        varied_words_tuples = [top_word_tuple, second_word_tuple]
        varied_words_embeds = [top_word_embed, second_word_embed]
        for _ in range(top_n - 2):
            # - Make sure we have at least one word available -
            if not sample_words:
                break
            # - Get the Info about the available words -
            sample_words_info = []
            for i in range(len(sample_words)):
                cur_word_tuple = sample_words[i]
                cur_word = cur_word_tuple[0]
                word_embed = word_embeds[cur_word]
                with_word_homo = mean_similarity(
                    vectors_ndarray=varied_words_embeds + [word_embed],
                    parallelism=True, show_progress=False
                )
                # Create Dictionary with the Info of the Word.
                cur_word_info = {
                    'word': cur_word,
                    'word_index': i,
                    'word_sim_tuple': cur_word_tuple,
                    'word_embed': word_embed,
                    'with_word_homo': with_word_homo,
                }
                # Save Info in list.
                sample_words_info.append(cur_word_info)
            # - Select the word with the lowest Homogeneity -
            best_varied_word = min(
                sample_words_info, key=lambda x_dict: x_dict['with_word_homo']
            )
            # - Update the Info about Selected Words -
            varied_words_tuples.append(best_varied_word['word_sim_tuple'])
            varied_words_embeds.append(best_varied_word['word_embed'])
            del sample_words[best_varied_word['word_index']]

        # Sort the Selected Word by their similarity to the Topic.
        varied_words_tuples.sort(key=lambda word_sim: word_sim[1], reverse=True)
        # The Group of Diverse Words that best describes the Topic.
        return varied_words_tuples

    def model_homogeneity(self, parallelism=True, show_progress=False):
        """
        Get the Homogeneity of the Topic Model. The sum of the homogeneity of
        the topics in the model.

        Args:
            parallelism: Bool indicating if we can use multiprocessing to speed
                up the execution of the program.
            show_progress: A Bool representing whether we show the progress of
                the method or not.
        Returns:
            Float with the Homogeneity of the Topic Model.
        """
        total_homogeneity = sum(
            self.topic_homogeneity(
                topic_id=topic_id, parallelism=parallelism, show_progress=show_progress
            )
            for topic_id in self.topic_ids
        )
        return total_homogeneity

    def cur_model_homogeneity(self, parallelism=True, show_progress=False):
        """
        Get the Homogeneity of the Current Topic Model. The sum of the
        homogeneity of the topics in the model.

        Args:
            parallelism: Bool indicating if we can use multiprocessing to speed
                up the execution of the program.
            show_progress: A Bool representing whether we show the progress of
                the method or not.
        Returns:
            Float with the Homogeneity of the Topic Model.
        """
        if show_progress:
            topics_homogeneity = []
            for cur_topic_id in self.cur_topic_ids:
                progress_msg(f"Calculating the Homogeneity of <{cur_topic_id}>...")
                new_homogeneity = self.cur_topic_homogeneity(
                    cur_topic_id=cur_topic_id, parallelism=parallelism,
                    show_progress=show_progress
                )
                topics_homogeneity.append(new_homogeneity)
            # Calculate the Sum of Topics' Homogeneity
            cur_total_homogeneity = sum(topics_homogeneity)
        else:
            cur_total_homogeneity = sum(
                self.cur_topic_homogeneity(
                    cur_topic_id=cur_topic_id, parallelism=parallelism,
                    show_progress=show_progress
                )
                for cur_topic_id in self.cur_topic_ids
            )
        return cur_total_homogeneity

    def calc_topics_homogeneity(self, parallelism=True, show_progress=False):
        """
        Calculate the homogeneity of all the original topics in the model and
        save them in the base_topics_homogeneity dictionary.
        """
        # Calculate the Homogeneity of the Topics.
        topics_homogeneity = {}
        for topic_id in self.topic_ids:
            if show_progress:
                progress_msg(f"Calculating Homogeneity of <{topic_id}>...")
            new_homogeneity = self._topic_homogeneity(
                topic_id=topic_id, parallelism=parallelism,
                show_progress=show_progress
            )
            # Save the new Homogeneity.
            topics_homogeneity[topic_id] = new_homogeneity
        # Save the Homogeneity of the topics in the Model.
        self.base_topics_homogeneity = topics_homogeneity

    def topic_homogeneity(
            self, topic_id: str, parallelism=True, show_progress=False
    ):
        """
        Get the Average similarity between the Documents of the Topic.
        """
        # Check if the Model has the Homogeneity Saved.
        if self.base_topics_homogeneity:
            # Use Previously Calculated Homogeneity.
            if show_progress:
                progress_msg("Using Previously calculated Homogeneity values.")
            return self.base_topics_homogeneity[topic_id]
        else:
            # Calculate the Homogeneity for the Topic.
            return self._topic_homogeneity(
                topic_id=topic_id, parallelism=parallelism,
                show_progress=show_progress
            )

    def _topic_homogeneity(
            self, topic_id: str, parallelism=True, show_progress=False
    ):
        """
        Get the Average similarity between the Documents of the Topic.
        """
        # Get the Embeddings of the Documents.
        doc_embeds = [
            self.base_doc_embeds[doc_id]
            for doc_id, _ in self.base_topic_docs[topic_id]
        ]
        # Find the Mean Similarity.
        mean_sim = mean_similarity(
            vectors_ndarray=doc_embeds, parallelism=parallelism,
            show_progress=show_progress
        )
        # The Homogeneity of the Topic - Average Similarity.
        return mean_sim

    def cur_topic_homogeneity(
            self, cur_topic_id: str, parallelism=True, show_progress=False
    ):
        """
        Get the Average similarity between the Document of the Current Topic
        'cur_topic_id'.
        """
        # Check if the current topics are the original topics.
        if not self.has_reduced_topics:
            return self.topic_homogeneity(
                topic_id=cur_topic_id, parallelism=parallelism,
                show_progress=show_progress
            )
        # Get the Embeddings of the Documents.
        doc_embeds = [
            self.base_doc_embeds[doc_id]
            for doc_id, _ in self.base_cur_topic_docs[cur_topic_id]
        ]
        # Find the Mean Similarity.
        mean_sim = mean_similarity(
            vectors_ndarray=doc_embeds, parallelism=parallelism,
            show_progress=show_progress
        )
        # The Homogeneity of the Topic - Average Similarity.
        return mean_sim

    def cur_topics_top_words(self, top_n=10, comparer='cos-sim'):
        """
        Get the 'n' top words that best describe each of the current topics in
        the model.
        """
        result_dict = {}
        for cur_topic_id, _ in self.cur_topic_by_size():
            result_dict[cur_topic_id] = self.top_words_cur_topic(
                cur_topic_id=cur_topic_id, top_n=top_n, comparer=comparer
            )
        # Dictionary with Top N words per current topic.
        return result_dict

    def model_pwi(self, word_num=-1, pwi_type='exact'):
        """
        Get the descriptive value of the topics of the model about the documents
        in the corpus using the PWI formula (exact or tf-idf).

        Args:
            pwi_type: String with the PWI formula to use 'exact' or 'tf-idf'.
            word_num: The number of words used to get the topics PWI.
        Returns:
            Float with the descriptive value of the topics of the model.
        """
        total_pwi = sum(
            self.topic_pwi(topic_id=topic_id, word_num=word_num, pwi_type=pwi_type)
            for topic_id in self.topic_ids
        )
        return total_pwi

    def cur_model_pwi(self, word_num=-1, pwi_type='exact'):
        """
        Get the descriptive value of the current topics of the model about the
        documents in the corpus using the PWI formula (exact or tf-idf).

        Args:
            pwi_type: String with the PWI formula to use 'exact' or 'tf-idf'.
            word_num: The number of words used to get the topics PWI.
        Returns:
            Float with the descriptive value of the current topics.
        """
        cur_total_pwi = sum(
            self.cur_topic_pwi(
                cur_topic_id=cur_topic_id, word_num=word_num, pwi_type=pwi_type
            )
            for cur_topic_id in self.cur_topic_ids
        )
        return cur_total_pwi

    def topic_pwi(self, topic_id: str, word_num=-1, pwi_type='exact'):
        """
        Get the topic's descriptive value for its documents using the mutual
        information formula.

        If 'word_num' is different from -1, then select the number 'word_num' of
        words closest to the topic, and get their PWI values.

        Args:
            topic_id: String with the ID of the topic.
            word_num: The number of words used to get the topic PWI.
            pwi_type: String with the PWI formula to use 'exact' or 'tf-idf'.
        Returns:
            Float with the PWI value of the Topic.
        """
        # Get the PWI value for all the words in the vocabulary of the topic.
        topic_words_pwi = self.topic_words_pwi(topic_id, pwi_type)
        # Check if we can use all the words in the topic.
        if word_num == -1 or word_num >= len(topic_words_pwi):
            topic_pwi = sum(topic_words_pwi.values())
        else:
            # Use the closest 'word_num' words in the semantic space.
            words_sims = self.base_topic_words[topic_id][:word_num]
            topic_pwi = sum(
                topic_words_pwi[word]
                for word, _ in words_sims
            )
        return topic_pwi

    def cur_topic_pwi(self, cur_topic_id: str, word_num=-1, pwi_type='exact'):
        """
        Get the current topic 'cur_topic_id' descriptive value for its documents
        using the mutual information formula.

        If 'word_num' is different from -1, then select the number 'word_num' of
        words closest to the topic, and get their PWI values.

        Args:
            cur_topic_id: String with the ID of the current topic.
            word_num: The number of words used to get the topic PWI.
            pwi_type: String with the PWI formula to use 'exact' or 'tf-idf'.
        Returns:
            Float with the PWI value of the current Topic.
        """
        # Get the PWI value for all the words in the vocabulary of the topic.
        cur_topic_words_pwi = self.cur_topic_words_pwi(cur_topic_id, pwi_type)
        # Check if we can use all the words in the topic.
        if word_num == -1 or word_num >= len(cur_topic_words_pwi):
            cur_topic_pwi = sum(cur_topic_words_pwi.values())
        else:
            # Use the closest 'word_num' words in the semantic space.
            words_sims = self.base_cur_topic_words[cur_topic_id][:word_num]
            cur_topic_pwi = sum(
                cur_topic_words_pwi[word]
                for word, _ in words_sims
            )
        return cur_topic_pwi

    def topic_words_pwi(self, topic_id: str, pwi_type='exact'):
        """
        Get the PWI of the words in the vocabulary of the topic.

        Args:
            topic_id: String with the ID of the topic.
            pwi_type: String with the PWI formula to use 'exact' or 'tf-idf'.
        Returns:
            Dictionary with the PWIs of each word in the topic.
        """
        # Get Topic's Attributes.
        word_list = [word for word, _ in self.base_topic_words[topic_id]]
        doc_list = [doc_id for doc_id, _ in self.base_topic_docs[topic_id]]
        corpus_vocab = self.base_corpus_vocab
        # Get the function to calculate the PWI (exact or tf-idf)
        if pwi_type == 'exact':
            pwi_func = corpus_vocab.word_pwi_exact
        elif pwi_type == 'tf-idf':
            pwi_func = corpus_vocab.word_pwi_tf_idf
        else:
            raise NameError(f"The Requested PWI type <{pwi_type}> is not supported.")
        # Get the PWI of the words of the topic.
        word_pwi_dict = dict(
            (word, pwi_func(word, doc_list)) for word in word_list
        )
        # Dictionary with the PWI for each word of the topic.
        return word_pwi_dict

    def cur_topic_words_pwi(self, cur_topic_id: str, pwi_type='exact'):
        """
        Get the PWI of the words in the vocabulary of the current topic
        'topic_id'.

        Args:
            cur_topic_id: String with the ID of a current topic.
            pwi_type: String with the PWI formula to use 'exact' or 'tf-idf'.
        Returns:
            Dictionary with the PWIs of each word in the topic.
        """
        # Get Current Topic's Attributes.
        word_list = [word for word, _ in self.base_cur_topic_words[cur_topic_id]]
        doc_list = [doc_id for doc_id, _ in self.base_cur_topic_docs[cur_topic_id]]
        corpus_vocab = self.base_corpus_vocab
        # Get the function to calculate the PWI (exact or tf-idf)
        if pwi_type == 'exact':
            pwi_func = corpus_vocab.word_pwi_exact
        elif pwi_type == 'tf-idf':
            pwi_func = corpus_vocab.word_pwi_tf_idf
        else:
            raise NameError(f"The Requested PWI type <{pwi_type}> is not supported.")
        # Get the PWI of the words of the topic.
        word_pwi_dict = dict(
            (word, pwi_func(word, doc_list)) for word in word_list
        )
        # Dictionary with the PWI for each word of the topic.
        return word_pwi_dict

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
            Dictionary with the Topic Embeddings of the new Reduced Topic Model.
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
            new_topic_embeds = self.base_topic_embeds.copy()
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
            json_topic_embeds = reduced_topic_index['topic_embeds']
            current_size = len(json_topic_embeds)
            new_topic_embeds = dict_list2ndarray(json_topic_embeds)
            new_topic_sizes = reduced_topic_index['topic_sizes']

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
            new_topic_embeds, new_topic_sizes = self.reduce_topic_size(
                ref_topic_embeds=new_topic_embeds, topic_sizes=new_topic_sizes,
                parallelism=parallelism, show_progress=show_progress
            )
            # Update Current Topic Size.
            current_size = len(new_topic_embeds)

        # Dictionary with the Topic Embeddings of the reduced Topic Model.
        if show_progress:
            progress_msg(f"Topic Model reduced to {current_size} topics.")
        return new_topic_embeds

    def base_save_reduced_topics(self, parallelism=False, override=False,
                                 show_progress=False):
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
        # Initialize Topic Reduction dictionaries.
        current_size = self.topic_size
        new_topic_embeds = self.base_topic_embeds.copy()
        new_topic_sizes = dict(
            [(topic_id, len(doc_list))
             for topic_id, doc_list in self.base_topic_docs.items()]
        )
        # Start Reducing Topics and Saving the Main Topic Sizes.
        if show_progress:
            progress_msg("Saving the Hierarchically Reduced Topic Models...")
        while current_size > 2:
            # Reduce the Topic Size by 1.
            if show_progress:
                progress_msg(
                    f"Reducing from {current_size} to {current_size - 1} topics..."
                )
            new_topic_embeds, new_topic_sizes = self.reduce_topic_size(
                ref_topic_embeds=new_topic_embeds, topic_sizes=new_topic_sizes,
                parallelism=parallelism, show_progress=show_progress
            )
            # Update the current number of topics.
            current_size = len(new_topic_embeds)

            # Check if we have to save the current embeddings and sizes.
            if current_size in main_sizes:
                if show_progress:
                    progress_msg(
                        "<<Main Topic Found>>\n"
                        f"Saving Reduced Topic Model with {current_size} topics..."
                    )
                # Transform Topic Embeddings to List[float].
                json_topic_embeds = dict_ndarray2list(new_topic_embeds)
                # Create Dict with embeddings and sizes.
                reduced_topics_index = {
                    'topic_embeds': json_topic_embeds,
                    'topic_sizes': new_topic_sizes,
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

    def reduce_topic_size(self, ref_topic_embeds: dict, topic_sizes: dict,
                          parallelism=False, show_progress=False):
        """
        Reduce by 1 the number of topics inside the dictionary 'ref_topic_embeds',
        joining the smallest topic with its closest neighbor.

        Args:
            ref_topic_embeds: Dictionary containing the embeddings of the topics
                we are going to reduce. This dictionary is treated as a
                reference and will be modified to store the new reduced topics.
            topic_sizes: Dictionary containing the current size of the topics we
                are reducing.
            parallelism: Bool to indicate if we have to use the multiprocessing
                version of this function.
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns:
            Tuple with 'ref_topic_embeds' dictionary  and a new 'topic_sizes'
                dictionary containing the updated embeddings and sizes
                respectively for the new Topics.
        """
        # Get the ID and Embed of the smallest Topic.
        min_topic_id = min(
            topic_sizes.keys(), key=lambda topic_id: topic_sizes[topic_id]
        )
        min_embed = ref_topic_embeds[min_topic_id]

        # Delete Smallest Topic.
        del ref_topic_embeds[min_topic_id]
        # Get the closest topic to the Smallest Topic.
        close_topic_id, _ = closest_vector(min_embed, ref_topic_embeds)
        close_embed = ref_topic_embeds[close_topic_id]
        # Merge the embeddings of the topics.
        min_size = topic_sizes[min_topic_id]
        close_size = topic_sizes[close_topic_id]
        total_size = min_size + close_size
        merged_topic_embed = (
                (min_size * min_embed + close_size * close_embed) / total_size
        )
        # Update embedding of the closest topic.
        ref_topic_embeds[close_topic_id] = merged_topic_embed

        # Get new the new Topic Sizes.
        if show_progress:
            progress_msg(f"Creating sizes for the new {len(ref_topic_embeds)} topics...")
        new_topic_sizes = count_child_embeddings(
            parent_embeds=ref_topic_embeds, child_embeds=self.base_doc_embeds,
            parallelism=parallelism, show_progress=show_progress
        )
        # Dictionaries with the new Topic Sizes and Embeddings.
        return ref_topic_embeds, new_topic_sizes

    def reduced_topics_saved(self):
        """
        Check if the given Topic Model created and saved the Hierarchically
        Reduced Topics.

        Returns: Bool showing if the Reduced Topic Models were saved.
        """
        # Check the Model's Folders.
        if not isdir(self.base_class_folder):
            return False
        model_folder_path = join(self.base_class_folder, self.model_folder_name)
        if not isdir(model_folder_path):
            return False
        reduced_folder_path = join(model_folder_path, self.base_reduced_folder)
        if not isdir(reduced_folder_path):
            return False

        # Check that all the Main Reduced Topic Models were saved.
        main_sizes = self.main_sizes(self.topic_size)
        for topic_size in main_sizes:
            # Check the file for the Reduced Model with the current size.
            reduced_topic_file = self.base_reduced_prefix + str(topic_size) + '.json'
            reduced_topic_path = join(reduced_folder_path, reduced_topic_file)
            if not isfile(reduced_topic_path):
                return False

        # All The Files were created correctly.
        return True

    @staticmethod
    def main_sizes(original_size: int):
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
    # noinspection PyUnresolvedReferences
    cluster_labels = clusters_found.labels_

    # Assign each cluster found to a new prominent topic.
    if show_progress:
        progress_msg("Assign each document in a cluster to a topic...")
    topic_clusters_embeds = {}
    for label, doc_embed in zip(cluster_labels, doc_embeds_list):
        # Skip Noise labels.
        if label == -1:
            continue
        # Check if this is the first time we found this prominent Topic.
        topic_id = 'Topic_' + str(label)
        if topic_id in topic_clusters_embeds:
            topic_clusters_embeds[topic_id].append(doc_embed)
        else:
            topic_clusters_embeds[topic_id] = [doc_embed]
    # Report number of topics.
    topic_size = len(topic_clusters_embeds)
    if show_progress:
        progress_msg(f"{topic_size} topics found.")

    # Refresh the Topic IDs (So they are correctly formatted)
    topic_clusters_embeds = refresh_topic_ids(topic_dict=topic_clusters_embeds)
    # # Report Prominent Topics cluster sizes.
    # if show_progress:
    #     progress_msg("<< Prominent Topics cluster sizes >>")
    #     for topic_id, cluster_embeds in topic_clusters_embeds.items():
    #         progress_msg(f" -> {topic_id}: {big_number(len(cluster_embeds))} docs")

    # Progress Variables.
    count = 0
    total = len(topic_clusters_embeds)
    # Create the topic's embeddings using the average of the doc's embeddings in
    # their cluster.
    topic_embeds = {}
    if show_progress:
        progress_msg(
            f"Creating the embeddings of the {topic_size} prominent topics..."
        )
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


def create_topic_words(topic_embeds: dict, topic_docs: dict, corpus_vocab: Vocabulary,
                       top_n=100, min_sim=0.0, show_progress=False):
    """
    Find the 'top_n' words that best describe each topic using the words from
    the documents of the topic.

    Args:
        topic_embeds: Dictionary(Topic ID -> Numpy.ndarray) with the
            embeddings of the Topics.
        topic_docs: Dictionary(Topic ID -> List[(str, float)]) with the list
            of documents belonging to each Topic and their similarity to the
            topic.
        corpus_vocab: Vocabulary created with the corpus used to create
            the Topic Model and the Document embeddings.
        top_n: Int with the amount of words we want to describe the topic.
        min_sim: The minimum value of cosine similarity accepted for a word
            to describe a Topic.
        show_progress: Bool representing whether we show the progress of
            the method or not.
    Returns:
        Dictionary(Topic ID -> List[str]) with the Topic IDs as keys and
            the list of words that best describe the topic as values.
    """
    # Get the Embedding of the Words.
    word_embeds = corpus_vocab.word_embeds

    # Progress Variables.
    count = 0
    total = len(topic_embeds)
    # Create Set of Words per each Topic.
    topic_top_words = {}
    for topic_id, topic_embed in topic_embeds.items():
        # Get the documents in the topic and their similarity.
        doc_sim_list = topic_docs[topic_id]
        # Create Set with the Words in the Documents.
        topic_vocab = {
            word for doc_id, _ in doc_sim_list
            for word in corpus_vocab.doc_words(doc_id)
        }
        # Get the similarity of the words to the topic.
        topic_words_sim = [
            (word, word_sim)
            for word in topic_vocab
            if min_sim < (word_sim := cosine_sim(topic_embed, word_embeds[word]))
        ]
        top_words = find_top_n(id_values=topic_words_sim, n=top_n, top_max=True)
        # Save the closest words.
        topic_top_words[topic_id] = top_words
        # Progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Dictionary with the Topics and their Top Words.
    return topic_top_words


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
    # See if we can use parallelism, checking the amount of iterations when
    # parallelism starts being faster than single core.
    parent_count = len(parent_embeds)
    child_count = len(child_embeds)
    total_iterations = child_count * parent_count
    if parallelism and total_iterations > BASE_ITERATIONS:
        return find_children_parallel(
            parent_embeds, child_embeds, show_progress=show_progress
        )

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
    total_iterations = len(parent_embeds) * len(child_embeds)
    efficiency_mult = min(float(1), total_iterations / PEAK_ITERATIONS)
    core_count = max(2, round(efficiency_mult * optimal_cores))
    # Create chunk size to process the tasks in the cores.
    chunk_size = max(1, len(child_embeds) // 100)

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
    find_children_parallel().

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


def count_child_embeddings(parent_embeds: dict, child_embeds: dict,
                           parallelism=False, show_progress=False):
    """
    Given the embedding's dictionaries 'parent_embeds' and 'child_embeds',
    create a new dictionary counting per each parent the number of child_embeds
    that are closest to them.

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
        Dictionary (Parent ID -> Int) containing the parent_ids as
            keys, and the number of child_ids that are closer to the Parent ID
            than any other parent.
    """
    # See if we can use parallelism, checking the amount of iterations when
    # parallelism starts being faster than single core.
    parent_count = len(parent_embeds)
    child_count = len(child_embeds)
    total_iterations = child_count * parent_count
    if parallelism and total_iterations > BASE_ITERATIONS:
        return count_children_parallel(
            parent_embeds, child_embeds, show_progress=show_progress
        )
    # Check we have at least one parent.
    if len(parent_embeds) == 0:
        return {}

    # Progress Variables.
    count = 0
    total = len(child_embeds)
    # Iterate through the children to find their closest parent.
    parent_child_count = {}
    for child_id, child_embed in child_embeds.items():
        # Find the closest parent to the child.
        parent_id, _ = closest_vector(child_embed, parent_embeds)
        # Check if we have found this parent before.
        if parent_id in parent_child_count:
            parent_child_count[parent_id] += 1
        else:
            parent_child_count[parent_id] = 1
        # Progress:
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Dictionary with Children count per each Parent.
    return parent_child_count


def count_children_parallel(parent_embeds: dict, child_embeds: dict, show_progress=False):
    """
    Version of count_child_embeddings() using parallelism.

    Given the embedding's dictionaries 'parent_embeds' and 'child_embeds',
    create a new dictionary counting per each parent the number of child_embeds
    that are closest to them.

    Args:
        parent_embeds: Dictionary containing the parent_ids as keys and their
            embeddings as values.
        child_embeds: Dictionary containing the child_ids as keys and their
            embeddings as values.
        show_progress: A Bool representing whether we show the progress of
            the method or not.
    Returns:
        Dictionary (Parent ID -> Int) containing the parent_ids as
            keys, and the number of child_ids that are closer to the Parent ID
            than any other parent.
    """
    # Check we have at least one Parent Dictionary.
    if len(parent_embeds) == 0:
        return {}

    # Determine the number of cores to be used. (I made my own formula)
    optimal_cores = min(multiprocessing.cpu_count(), MAX_CORES)
    total_iterations = len(parent_embeds) * len(child_embeds)
    efficiency_mult = min(float(1), total_iterations / PEAK_ITERATIONS)
    core_count = max(2, round(efficiency_mult * optimal_cores))
    # Create chunk size to process the tasks in the cores.
    chunk_size = max(1, len(child_embeds) // 100)

    # Create tuple parameters.
    tuple_params = [
        (child_embed, parent_embeds) for child_embed in child_embeds.values()
    ]
    # Create the Parent -> Child Count dictionary.
    parent_child_count = {}
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
                _count_closest_vector, tuple_params, chunksize=chunk_size
            )
            # Add count to the child_ids' closest parent.
            for parent_id in imap_results_iter:
                # Check if we have found this parent before.
                if parent_id in parent_child_count:
                    parent_child_count[parent_id] += 1
                else:
                    parent_child_count[parent_id] = 1
                # Report Progress.
                count += 1
                progress_bar(count, total)
        # No need to report progress, use Pool.map()
        else:
            # Get the closest parent to each child.
            map_results = pool.map(
                _count_closest_vector, tuple_params, chunksize=chunk_size
            )
            # Get the Number of Children per each Parent.
            for parent_id in map_results:
                if parent_id in parent_child_count:
                    parent_child_count[parent_id] += 1
                else:
                    parent_child_count[parent_id] = 1

    # Dictionary with Children count per each Parent.
    return parent_child_count


def _count_closest_vector(embed_parents: tuple):
    """
    Custom-made version of the method closest_vector() to use with the method
    count_children_parallel(), given that Pool.map & Pool.imap can take
    iterables the multiple parameters.

    In this version we don't return the similarity between the 'child_embed' and
    the closest parent, the method count_children_parallel() doesn't need it.

    Args:
        embed_parents: Tuple with (child_embed, parent_embeds).
    Returns:
        String with the ID of the closest parent.
    """
    child_embed, parent_embeds = embed_parents
    parent_id, _ = closest_vector(child_embed, parent_embeds)
    return parent_id


def refresh_topic_ids(topic_dict: dict):
    """
    Updates the IDs (keys) of the Topics in the dictionary, naming the Topics
    'Topic_1', 'Topic_2', ..., 'Topic_N' where N is the number of topics in the
    dictionary.
    The <N> number is formatted so all topics numbers have the same string
    length. For example, if the dictionary has 222 topics, then the topic IDs
    are formatted as 'Topic_001', ..., 'Topic_012'..., 'Topic_222'.

    Returns: Dictionary with the Dict_Keys updated.
    """
    # Get Topic Size.
    size_total = len(str(len(topic_dict)))
    topic_values = list(topic_dict.values())
    id_count = 0
    new_topic_dict = {}
    for t_value in topic_values:
        id_count += 1
        new_id = 'Topic_' + number_to_size(number=id_count, size=size_total)
        new_topic_dict[new_id] = t_value

    # Dictionary with new Topic IDs.
    return new_topic_dict
