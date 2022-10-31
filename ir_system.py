# Gelin Eguinosa Rosique
# 2022

from random import choice
from numpy import ndarray

from mono_topics import MonoTopics
from mix_topics import MixTopics
from corpora_manager import CorporaManager
from sample_manager import SampleManager
from model_manager import ModelManager
from util_funcs import cosine_sim, find_top_n
from extra_funcs import progress_msg

# Testing Imports.
import sys
from pprint import pprint
from time_keeper import TimeKeeper


class IRSystem:
    """
    Class to search information in the corpus (Cord-19) using one of the Topic
    Models.
    """

    def __init__(self, model_name: str, show_progress=False):
        """
        Using the provided 'topic_model' load the corpus and the text model of
        the topic model to be able to encode the queries made by the user and
        extract the most relevant documents.

        Args:
            model_name: String with the name of the Topic Model we have to use
                in the IR system.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Create Topic Model.
        mono_topics_ids = MonoTopics.saved_models()
        mix_topics_ids = MixTopics.saved_models()
        # Load the given 'model_name'.
        if show_progress:
            progress_msg(f"Loading Topic Model <{model_name}>...")
        if model_name in mono_topics_ids:
            topic_model = MonoTopics.load(
                model_id=model_name, show_progress=show_progress
            )
        elif model_name in mix_topics_ids:
            topic_model = MixTopics.load(
                model_id=model_name, show_progress=show_progress
            )
        else:
            # Model not available.
            raise FileNotFoundError(
                f"The Topic Model <{model_name}> is either not available or not"
                f"supported."
            )

        # Create a Corpus with the IDs of the documents in the Topic Model.
        if show_progress:
            progress_msg("Creating Corpus for the IR system...")
        # Get the Docs in the Topic Model and create main corpus.
        corpus_id = topic_model.base_corpus_id
        doc_ids = list(topic_model.base_doc_embeds.keys())
        main_corpus = CorporaManager(corpus_id=corpus_id, show_progress=show_progress)
        # Check if the Documents are the whole corpus.
        if len(doc_ids) == len(main_corpus):
            corpus = main_corpus
        else:
            # They are a Sample of the Corpus.
            corpus = SampleManager.load_custom(
                corpus=main_corpus, doc_ids=doc_ids, show_progress=show_progress
            )

        # Load the Text Model of the Topic Model.
        if show_progress:
            progress_msg("Loading Text Model for the IR system...")
        text_model_name = topic_model.base_text_model_name
        text_model = ModelManager(
            model_name=text_model_name, show_progress=show_progress
        )

        # Supported Topic Models.
        self.mono_topics_ids = mono_topics_ids
        self.mix_topics_ids = mix_topics_ids
        # Topic Model Attributes.
        self.model_name = model_name
        self.topic_model = topic_model
        # Corpus Attributes.
        self.main_corpus = main_corpus
        self.corpus = corpus
        # Text Model Attributes.
        self.text_model = text_model

    @property
    def topic_size(self):
        """
        Get the current size of the Topic Model.
        """
        return self.topic_model.cur_topic_size

    @property
    def system_doc_ids(self):
        """
        IDs of the documents in the Topic Model of the System.
        """
        return self.corpus.doc_ids

    def update_model(self, new_model: str, show_progress=False):
        """
        Update the Topic Model used by the IR system.

        Args:
            new_model: String with the name of the new Topic Model to be used.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check we have a new model name.
        if new_model == self.model_name:
            # No need to update.
            return

        # Load the New Topic Model.
        if show_progress:
            progress_msg(f"Loading the Topic Model <{new_model}>...")
        if new_model in self.mono_topics_ids:
            topic_model = MonoTopics.load(
                model_id=new_model, show_progress=show_progress
            )
        elif new_model in self.mix_topics_ids:
            topic_model = MixTopics.load(
                model_id=new_model, show_progress=show_progress
            )
        else:
            # The New Model is not available.
            raise FileNotFoundError(
                f"The Topic Model <{new_model}> is either not available or not"
                f"supported."
            )

        # Updating the Corpus of the IR system.
        if show_progress:
            progress_msg("Loading New Corpus for the IR system...")
        # Get the Info about the corpus in the Topic Model.
        corpus_id = topic_model.base_corpus_id
        doc_ids = list(topic_model.base_doc_embeds.keys())
        # Check if we have to upload a new Main corpus.
        if corpus_id != self.main_corpus.corpus_id:
            main_corpus = CorporaManager(
                corpus_id=corpus_id, show_progress=show_progress
            )
        else:
            # No need to change the Main Corpus.
            if show_progress:
                progress_msg(f"Using the same Main Corpus <{corpus_id}>")
            main_corpus = self.main_corpus
        # Check if the documents represent the whole corpus or a sample.
        if len(doc_ids) == len(main_corpus):
            corpus = main_corpus
        else:
            # We have to use a Sample of the Corpus.
            corpus = SampleManager.load_custom(
                corpus=main_corpus, doc_ids=doc_ids, show_progress=show_progress
            )

        # Check if we have to update the Text Model.
        cur_text_model_name = self.text_model.model_name
        new_text_model_name = topic_model.base_text_model_name
        if cur_text_model_name != new_text_model_name:
            if show_progress:
                "Loading a New Text Model ..."
            text_model = ModelManager(
                model_name=new_text_model_name, show_progress=show_progress
            )
        else:
            # Using the Same Text Model.
            text_model = self.text_model

        # Update Topic Model Attributes.
        self.model_name = new_model
        self.topic_model = topic_model
        # Update Corpus Attributes.
        self.main_corpus = main_corpus
        self.corpus = corpus
        # Text Model Attributes.
        self.text_model = text_model

    def update_topic_size(self, new_size: str, show_progress=False):
        """
        Update the Topic Model Size to one of the Supported Sizes of the Model.
        """
        # Transform the String to Int.
        new_topic_size = int(new_size)
        # Reduce the Size of the Topic Model.
        self.topic_model.reduce_topics(
            new_size=new_topic_size, parallelism=True, show_progress=show_progress
        )
        if show_progress:
            progress_msg(f"The IR System has a new Topic Size <{new_topic_size}>.")

    def supported_model_sizes(self):
        """
        Create a list with the sizes of the saved Reduced Topic Models (usually
        the main sizes).

        Returns: List[int] with the saved reduced sizes.
        """
        # Check if the model has reduced sizes.
        if self.topic_model.reduced_topics_saved():
            saved_sizes = list(self.topic_model.main_sizes(self.topic_model.topic_size))
            saved_sizes.sort()
            saved_sizes.append(self.topic_model.topic_size)
        else:
            # Only the Original Size of the Model.
            saved_sizes = [self.topic_model.topic_size]
        # Saved Topic Sizes.
        return saved_sizes

    def user_query(self, query: str, topic_num=10, doc_num=10, show_progress=False):
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
        # Get the Embedding of the Query.
        if show_progress:
            progress_msg("Encoding the text of the Query...")
        query_embed = self.text_model.doc_embed(doc=query)
        # Use the 'embed_query' method.
        top_topics_sims, top_docs_sims = self.embed_query(
            embed=query_embed, topic_num=topic_num, doc_num=doc_num,
            show_progress=show_progress
        )
        # Most Relevant Topics and Documents with their similarity.
        return top_topics_sims, top_docs_sims

    def text_embed(self, text: str):
        """
        Given the text of a query, create the embedding of the query using the
        Text Model of the Topic Model. This vector representation would be in
        the vector space that contains words, topics and documents.
        """
        query_embed = self.text_model.doc_embed(doc=text)
        return query_embed

    def embed_query(
            self, embed: ndarray, topic_num=10, doc_num=10,
            vector_space='words', show_progress=False
    ):
        """
        Given an Embedding 'embed', find the closest topics and documents to the
        embedding using the Topic Model. Similar to 'self.user_query()'
        """
        # Find The Top Topics For the Embedding.
        if show_progress:
            progress_msg("Finding the most relevant topics...")
        # Get the Embeddings of the Topics.
        if vector_space == 'words':
            topic_embeds = self.topic_model.current_word_space_topic_embeds
        elif vector_space == 'documents':
            topic_embeds = self.topic_model.current_doc_space_topic_embeds
        else:
            raise NameError(f"The Vector Space <{vector_space}> is not supported.")
        # Calculate the distance of the topics to the embeds.
        topics_sims = [
            (topic_id, cosine_sim(embed, topic_embed))
            for topic_id, topic_embed in topic_embeds.items()
        ]
        # Get the Most Relevant Topics.
        top_topics_sims = find_top_n(id_values=topics_sims, n=topic_num)
        top_topic_id, _ = top_topics_sims[0]

        # Get the IDs of the Documents in the Top Topic.
        doc_ids = self.topic_doc_ids(top_topic_id)
        # Get the Most Relevant Documents.
        top_docs_sims = self.embed_query_top_docs(
            embed=embed, doc_ids=doc_ids, doc_num=doc_num,
            vector_space=vector_space, show_progress=show_progress
        )

        # Most Relevant Topics and Documents with their similarity.
        return top_topics_sims, top_docs_sims

    def embed_query_top_docs(
            self, embed: ndarray, doc_ids=None, doc_num=10,
            vector_space='words', show_progress=False
    ):
        """
        Given an Embedding 'embed', find the closest documents in the provided
        list 'doc_ids' to the given embed. If 'doc_ids' is empty, the use all
        the documents in the Topic Model.
        """
        if show_progress:
            progress_msg("Finding the most relevant docs...")

        # Check the List of Document IDs is not empty.
        if not doc_ids:
            doc_ids = list(self.topic_model.doc_space_doc_embeds.keys())

        # Get the Doc Embeddings.
        if vector_space == 'words':
            doc_embeds = self.topic_model.word_space_doc_embeds
        elif vector_space == 'documents':
            doc_embeds = self.topic_model.doc_space_doc_embeds
        else:
            raise NameError(f"The Vector Space <{vector_space}> is not supported.")

        # Calculate the Distance of the Documents to the Embedding.
        docs_sims = iter(
            (doc_id, cosine_sim(embed, doc_embeds[doc_id]))
            for doc_id in doc_ids
        )
        # Get the Most Relevant Documents.
        doc_count = len(doc_ids)
        find_progress = doc_count >= 10_000
        top_docs_sims = find_top_n(
            id_values=docs_sims, n=doc_num, iter_len=doc_count,
            show_progress=find_progress
        )

        # Most Relevant Documents for the given 'embed'.
        return top_docs_sims

    def topic_vocab(self, topic_id: str):
        """
        Get the list of words that best describe the topic and their similarity
        to the topic.
        """
        words_sims_list = self.topic_model.base_cur_topic_words[topic_id]
        return words_sims_list

    def topic_doc_ids(self, topic_id: str):
        """
        Get the Document IDs of the Topic 'topic_id'.
        """
        doc_ids = [
            doc_id
            for doc_id, _ in self.topic_model.base_cur_topic_docs[topic_id]
        ]
        return doc_ids

    def topics_info(self, sort_cat='size', word_num=10, use_varied_vocab=False):
        """
        Create a list of tuples with the information about the current topics
        in the Topic Model, including:
         - Topic ID
         - Value for the given Sorting Category.
         - Description (top words that best describe them).

        The Topics can be sorted by 3 types of categories their Size, PWI-tf-idf
        or PWI-exact. The value of the second element in the tuple of each topic
        will depend on the selected category.

        The PWI value of the Topics will depend on the words of the topics
        returned, if 5 words were requested, those 5 words will be used for the
        descriptive value of their topic.

        Args:
            sort_cat: String with the category of the parameters to rank the
                topics in the list.
            word_num: Int with the number of words in the description of the
                Topic.
            use_varied_vocab: Indicates if we are going to use a Varied Vocab to
                describe the topic.
        Returns:
            List[Tuple(ID, Category_Value, Description)] with the info of the
                topics.
        """
        # Get the Current Topics.
        if sort_cat == 'size':
            topics_cat_values = self.topic_model.cur_topic_by_size()
        elif sort_cat == 'homogeneity':
            topics_cat_values = self.topic_model.cur_topic_by_homogeneity()
        elif sort_cat in {'pwi-tf-idf', 'pwi-exact'}:
            pwi_type = sort_cat[4:]
            topics_cat_values = self.topic_model.cur_topic_by_pwi(
                word_num=word_num, pwi_type=pwi_type
            )
        else:
            raise NameError("The Topic Model does not support the sorting "
                            f"category <{sort_cat}>.")
        # Add the Descriptions of the Topics to the List.
        topics_info = [
            (topic_id, cat_value,
             self.topic_description(
                 topic_id=topic_id, word_num=word_num,
                 use_varied_vocab=use_varied_vocab
             ))
            for topic_id, cat_value in topics_cat_values
        ]
        # List with the Topics Info: (ID, Size, Description)
        return topics_info

    def topic_docs_info(self, topic_id: str):
        """
        Create a list of tuples with the information about the Documents that
        belong to the given 'topic_id', including:
         - Doc ID
         - Similarity
         - Title
         - Abstract

        Args:
            topic_id: String with the ID of the topic.
        Returns:
            List[Tuple(ID, Sim, Title, Abstract)] with the info of the documents
                in the topic.
        """
        topic_docs = self.topic_model.base_cur_topic_docs[topic_id]
        docs_info = [
            (doc_id, similarity,
             self.corpus.doc_title(doc_id), self.corpus.doc_abstract(doc_id))
            for doc_id, similarity in topic_docs
        ]
        # List with the Doc's Info: (ID, Similarity, Title, Abstract)
        return docs_info

    def topic_description(self, topic_id: str, word_num=10, use_varied_vocab=False):
        """
        Create a string with the words that best describe the topic 'topic_id'.

        Args:
            topic_id: String with the ID of the topic.
            word_num: Int with the number of words we want to describe the word.
            use_varied_vocab: Indicates if we are going to use a Varied Vocab to
                describe the topic.
        Returns:
            String with the words that describe the topic.
        """
        # Check if we have to create a Varied Description.
        if use_varied_vocab:
            top_words = self.topic_model.cur_topic_varied_words(
                cur_topic_id=topic_id, top_n=word_num
            )
        # Use the Closest Words to the Topic.
        else:
            top_words = self.topic_model.top_words_cur_topic(
                cur_topic_id=topic_id, top_n=word_num
            )
        # Create String with the description.
        first_word, _ = top_words[0]
        descript_text = first_word
        for word, _ in top_words[1:]:
            descript_text += ', ' + word
        # Text with the description of the Topic.
        return descript_text

    def random_doc_id(self):
        """
        Select a Random Document from the Corpus of the Topic Model and return
        its ID.
        """
        rand_doc_id = choice(list(self.topic_model.base_doc_embeds.keys()))
        return rand_doc_id

    def doc_embed(self, doc_id: str, space='documents'):
        """
        Get the Embedding of the Document 'doc_id'.
        """
        if space == 'documents':
            doc_embed = self.topic_model.doc_space_doc_embeds[doc_id]
        elif space == 'words':
            doc_embed = self.topic_model.word_space_doc_embeds[doc_id]
        else:
            raise NameError(f"The Vector Space <{space}> is not supported.")
        return doc_embed

    def doc_title(self, doc_id: str):
        """
        Get the Title of the Document 'doc_id'.
        """
        title = self.corpus.doc_title(doc_id)
        return title

    def doc_abstract(self, doc_id: str):
        """
        Get the Abstract of the Document 'doc_id'.
        """
        abstract = self.corpus.doc_abstract(doc_id)
        return abstract

    def doc_full_content(self, doc_id: str):
        """
        Get the Full Content of the Document 'doc_id'. If the Document has no
        content it returns the empty string ('').
        """
        full_content = self.corpus.doc_body_text(doc_id)
        return full_content

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

    def is_original_size(self):
        """
        Check if the Topic Model of the IR System has currently the original
        size of the model a.k.a. it has the maximum number of topics possible.
        """
        return not self.topic_model.has_reduced_topics


if __name__ == '__main__':
    # Record the Runtime of the Program.
    _stopwatch = TimeKeeper()
    # Terminal Parameters.
    _args = sys.argv

    # Create IR system.
    print(f"\nCreating the IR System instance...")
    _topic_model_id = 'sbert_fast_20_000_docs_182_topics'
    _ir_sys = IRSystem(model_name=_topic_model_id, show_progress=True)
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
            _topic_descript = _ir_sys.topic_description(
                topic_id=_topic_id, word_num=10, use_varied_vocab=True
            )
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
