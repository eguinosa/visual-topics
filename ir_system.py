# Gelin Eguinosa Rosique
# 2022

from base_topics import BaseTopics
from corpora_manager import CorporaManager
from sample_manager import SampleManager
from model_manager import ModelManager
from util_funcs import cosine_sim, find_top_n
from extra_funcs import progress_msg

# Testing Imports.
import sys
from pprint import pprint
from mono_topics import MonoTopics
from time_keeper import TimeKeeper


class IRSystem:
    """
    Class to search information in the corpus (Cord-19) using one of the Topic
    Models.
    """

    def __init__(self, topic_model: BaseTopics, show_progress=False):
        """
        Using the provided 'topic_model' load the corpus and the text model of
        the topic model to be able to encode the queries made by the user and
        extract the most relevant documents.

        Args:
            topic_model: Model used to create the topics in the corpus.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Create a Corpus with the IDs of the documents in the Topic Model.
        if show_progress:
            progress_msg("Creating Corpus for the IR system...")
        corpus_id = topic_model.base_corpus_id
        doc_ids = list(topic_model.base_doc_embeds.keys())
        main_corpus = CorporaManager(corpus_id=corpus_id, show_progress=show_progress)
        corpus = SampleManager.load_custom(
            corpus=main_corpus, doc_ids=doc_ids, show_progress=show_progress
        )
        # Load the Text Model of the Topic Model.
        if show_progress:
            progress_msg("Loading Text Model for the IR system...")
        text_model_name = topic_model.base_text_model_name
        text_model = ModelManager(model_name=text_model_name, show_progress=show_progress)

        # Save the class Attributes.
        self.topic_model = topic_model
        self.corpus = corpus
        self.text_model = text_model

    def user_query(self, query: str, topic_num=10, doc_num=10, show_progress=True):
        """
        Given a user's 'query' get the top 'doc_num' of relevant documents and
        the top 'topic_num' of relevant topics for the query. The top documents
        will belong only to most relevant topic. If the user would like to
        include documents from other relevant topics, they would need to use
        query expansion after the results have been given.

        Args:
            query: String with the text of the query made by the user.
            topic_num: Int with the number of relevant documents we have to
                return to the user.
            doc_num: Int with the number of relevant documents we have to return
                to the user.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        Returns:
            List[(Topic_ID, Sim)], List[(Doc_ID, Sim)] with the list of the
                most relevant topics to the query and their similarity, and the list
                of the most relevant documents and their similarity from the most
                relevant topic.
        """
        # Get the topic embeds from the vocabulary vector space.
        topic_embeds = self.topic_model.current_word_space_topic_embeds

        # Get the Embedding of the Query.
        if show_progress:
            progress_msg("Finding the most relevant topics...")
        query_embed = self.text_model.doc_embed(doc=query)
        topics_sims = [
            (topic_id, cosine_sim(query_embed, topic_embed))
            for topic_id, topic_embed in topic_embeds.items()
        ]
        # Get the Most Relevant Topics.
        top_topics_sims = find_top_n(id_values=topics_sims, n=topic_num)
        top_topic_id, _ = top_topics_sims[0]

        # Get the Embeddings of the Documents in the top topic.
        if show_progress:
            progress_msg("Finding most relevant docs in the top topic...")
        docs_ids = self.topic_model.base_topic_docs[top_topic_id]
        doc_embeds = self.topic_model.word_space_doc_embeds
        docs_sims = [
            (doc_id, cosine_sim(query_embed, doc_embeds[doc_id]))
            for doc_id, _ in docs_ids
        ]
        # Get the Most Relevant Documents.
        top_docs_sims = find_top_n(id_values=docs_sims, n=doc_num)

        # Most Relevant Topics and Documents with their similarity.
        return top_topics_sims, top_docs_sims

    def topic_description(self, topic_id: str, word_num=10):
        """
        Create a string with the words that best describe the topic 'topic_id'.

        Args:
            topic_id: String with the ID of the topic.
            word_num: Int with the number of words we want to describe the word.
        Returns:
            String with the words that describe the topic.
        """
        # Get the Top Words.
        top_words = self.topic_model.top_words_topic(
            topic_id=topic_id, top_n=word_num
        )
        # Create String with the description.
        first_word, _ = top_words[0]
        descript_text = first_word
        for word, _ in top_words[1:]:
            descript_text += ', ' + word
        # Text with the description of the Topic.
        return descript_text

    def doc_content(self, doc_id: str, len_limit=-1):
        """
        Get the formatted content of the document 'doc_id', with title and
        abstract. If the 'len_limit' parameter is different from -1, then cut
        the size of the content to fit the given size.

        Args:
            doc_id: String with the ID of the document.
            len_limit: Int with the char size limit we want the content to have.
        Returns:
            String with the formatted Title and Abstract of the Document.
        """
        # Get title and abstract.
        doc_title = self.corpus.doc_title(doc_id)
        doc_abstract = self.corpus.doc_abstract(doc_id)
        # Format the content.
        doc_content = (
            "Title: " + doc_title +
            "\n\n" +
            "Abstract:\n" +
            doc_abstract
        )
        # Check if we have to limit the size.
        if len_limit > 0:
            doc_content = doc_content[:len_limit] + "..."
        # Formatted content of the document.
        return doc_content


if __name__ == '__main__':
    # Record the Runtime of the Program.
    _stopwatch = TimeKeeper()
    # Terminal Parameters.
    _args = sys.argv

    # Load Topic Model.
    _loading_id = 'sbert_fast_20_000_docs_182_topics'
    print(f"\nLoading Topic Model with ID <{_loading_id}>...")
    _loaded_model = MonoTopics.load(model_id=_loading_id, show_progress=True)
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")

    # Create IR system.
    print(f"\nCreating the IR System instance...")
    _ir_sys = IRSystem(topic_model=_loaded_model, show_progress=True)
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")

    # Test Searching info in the Corpus.
    while True:
        # Make a Query.
        _query = input("\nSearch: ")
        if _query in {'', 'q', 'quit'}:
            break
        # Search in the IR System.
        _result_num = 5
        print("\nSearching in the corpus...")
        _topics_sims, _docs_sims = _ir_sys.user_query(
            query=_query, topic_num=_result_num, doc_num=_result_num,
            show_progress=True
        )
        # Relevant Topics.
        print(f"\n<<<--- Top {_result_num} Topics in search --->>>")
        _count = 0
        for _topic_id, _sim in _topics_sims:
            _topic_descript = _ir_sys.topic_description(topic_id=_topic_id, word_num=10)
            _count += 1
            print(f"\n-- Result {_count} --")
            print(f"{_topic_id} ({round(_sim, 3)} similar):")
            pprint(_topic_descript)
        # Relevant Documents.
        print(f"\n<<<--- Top {_result_num} Document in search --->>>")
        _count = 0
        for _doc_id, _sim in _docs_sims:
            _doc_content = _ir_sys.doc_content(doc_id=_doc_id, len_limit=250)
            _count += 1
            print(f"\n-- Result {_count} --")
            print(f"Doc<{_doc_id}> ({round(_sim, 3)} similar):")
            pprint(_doc_content)

    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
