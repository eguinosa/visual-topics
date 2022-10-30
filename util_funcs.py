# Gelin Eguinosa Rosique
# 2022

import multiprocessing
import numpy as np
from numpy.linalg import norm
from bisect import bisect

from extra_funcs import progress_bar, progress_msg

# Test imports.
import sys
# import random
# from pprint import pprint
from time_keeper import TimeKeeper


# The Core Multiplier to calculate the Chunk sizes when doing Parallelism.
PEAK_ITERATIONS = 3_750_000  # 150 topics * 25,000 docs
BASE_ITERATIONS = 925_000  # 37 topics * 25,000 docs
MAX_CORES = 8


def dict_ndarray2list(embeds_dict: dict, show_progress=False):
    """
    Transform the values of the dictionary 'embeds_dict' from Numpy.ndarray to
    list.

    Args:
        embeds_dict: Dictionary(ID -> List[Numpy.ndarray]) with the embeddings
            we want to transform to list.
        show_progress: Bool representing whether we show the progress of
            the method or not.
    Returns:
        Dictionary(ID -> List[Float]) with the embeddings of the dictionary
            with type list instead of Numpy.ndarray.
    """
    # Progress Variables.
    count = 0
    total = len(embeds_dict)
    # Default Dict.
    new_embeds_dict = {}
    for item_id, embed in embeds_dict.items():
        new_embeds_dict[item_id] = embed.tolist()
        if show_progress:
            count += 1
            progress_bar(count, total)
    # The Transformed Dictionary.
    return new_embeds_dict


def dict_list2ndarray(embeds_dict: dict, show_progress=False):
    """
    Transform the values of the dictionary 'embeds_dict' from List to
    Numpy.ndarray.

    Args:
        embeds_dict: Dictionary(ID -> List[Float]) with the embeddings we want
            to transform to Numpy.ndarray.
        show_progress: Bool representing whether we show the progress of
            the method or not.
    Returns:
        Dictionary(ID -> List[Numpy.ndarray]) with the embeddings of the
            dictionary with type Numpy.ndarray instead of List.
    """
    # Progress Variables.
    count = 0
    total = len(embeds_dict)
    # Default Dict.
    new_embeds_dict = {}
    for item_id, embed in embeds_dict.items():
        new_embeds_dict[item_id] = np.array(embed)
        if show_progress:
            count += 1
            progress_bar(count, total)
    # Transformed Dictionary.
    return new_embeds_dict


def cosine_sim(a: np.ndarray, b: np.ndarray):
    """
    Calculate the cosine similarity between the vectors 'a' and 'b'.

    Args:
        a: Numpy.ndarray containing one of the vectors embeddings.
        b: Numpy.ndarray containing one of the vectors embeddings.

    Returns:
        Float with the cosine similarity between the two vectors.
    """
    # Use Numpy.
    result = np.dot(a, b) / (norm(a) * norm(b))
    # Transform from float32 to float (float32 is not JSON serializable)
    result = float(result)
    return result


def closest_vector(embedding, vectors_dict: dict):
    """
    Given the embedding of a document or word and a dictionary containing a
    group of IDs with their embeddings. Find the closest vector to the given
    embedding using cosine similarity.

    Args:
        embedding: Numpy.ndarray with the embedding of the word or document that
            we are using to find the closest vector.
        vectors_dict: Dictionary containing the vectors with their IDs as
            keys and their embeddings as values.

    Returns:
        A tuple with the ID of the closest vector and its similarity to the
            'embedding' we received as parameter.
    """
    # Use iter to get the vectors IDs and their embeddings.
    vector_iter = iter(vectors_dict.items())

    # Get cosine similarity to the first vector.
    closest_vector_id, vector_embed = next(vector_iter)
    max_similarity = cosine_sim(embedding, vector_embed)

    # Iterate through the rest of the vectors.
    for vector_id, vector_embed in vector_iter:
        new_similarity = cosine_sim(embedding, vector_embed)
        if new_similarity > max_similarity:
            # New Closer Vector
            closest_vector_id = vector_id
            max_similarity = new_similarity

    # The closest vector ID with its similarity to the 'embedding'.
    return closest_vector_id, max_similarity


def find_top_n(id_values: iter, n=50, top_max=True, iter_len=-1, show_progress=False):
    """
    Given a list of tuples (IDs, Values) find the top 'n' tuples in the list
    using their values.

    Args:
        id_values: List[Tuple(IDs, Values)] with the IDs and Values we want to
            use to get the top 'n' values.
        n: Int with the amount of Top IDs we are going to return.
        top_max: Bool indicating the Maximum value is the Top value when 'True',
            and the Minimum value is the Top value when 'False'.
        iter_len: Int with the length of the iterable sequence to be able to
            show the progress of the function.
        show_progress: Bool representing whether we show the progress of
            the method or not.
    Returns:
        List[Tuple(IDs, Values)] with the top 'n' tuples in the list.
    """
    # Top Values list.
    top_ids_values = []
    sort_values = []

    # Check the value of iter_len.
    if iter_len == -1 and show_progress:
        # If the 'id_values' has a length attributes no problems.
        if hasattr(id_values, '__len__'):
            iter_len = len(id_values)
        else:
            # We can't show the progress.
            show_progress = False

    # Iterate though the elements of the dictionary to get the top n.
    count = 0
    total = iter_len
    for item_id, value in id_values:
        # Check if we want the top maximum values, if that's the case multiply
        # by -1 so the max values are now the minimum values.
        if top_max:
            sort_value = -1 * value
        else:
            sort_value = value
        # Fill the list with values first.
        if len(top_ids_values) < n:
            index = bisect(sort_values, sort_value)
            top_ids_values.insert(index, (item_id, value))
            sort_values.insert(index, sort_value)
        # If the list is full, check if check is this value is better.
        elif len(top_ids_values) >= n and sort_value < sort_values[-1]:
            # Delete last element of the lists.
            del top_ids_values[-1]
            del sort_values[-1]
            # Add the current value to the sorted lists.
            index = bisect(sort_values, sort_value)
            top_ids_values.insert(index, (item_id, value))
            sort_values.insert(index, sort_value)
        # Progress of the search.
        if show_progress:
            # Check the length is correct:
            if count > total:
                raise IndexError(f"More than {iter_len} items found.")
            count += 1
            progress_bar(count, total)

    # The Top tuples of (IDs, Values).
    return top_ids_values


def mean_similarity(vectors_ndarray: list, parallelism=False, show_progress=False):
    """
    Find the average similarity between the vectors inside the given vector
    cluster/group 'vectors_ndarray'.

    Args:
        vectors_ndarray: List[Numpy.ndarray] with the vector inside the cluster.
        parallelism: Bool indicating if we can use multiprocessing to speed up
            the execution of the program.
        show_progress: A Bool representing whether we show the progress of
            the method or not.
    Returns:
        Float between 0.0 and 1.0 with the average similarity between the
            vectors.
    """
    # Check we have enough vectors.
    if len(vectors_ndarray) < 2:
        return 1.0

    # See if we can use parallelism.
    total_iterations = (len(vectors_ndarray) * (len(vectors_ndarray) - 1)) // 2
    if parallelism and total_iterations > BASE_ITERATIONS:
        return mean_similarity_parallel(vectors_ndarray, show_progress)

    # Progress Variables.
    # Add the action of calculating the average calculation to total.
    count = 0
    total = total_iterations + 1

    # Simpler code if we don't have to show the progress.
    if show_progress:
        # Get the Similarities between the vectors.
        vector_similarities = []
        for i in range(1, len(vectors_ndarray)):
            for j in range(i):
                # Calculate Similarity.
                new_sim = cosine_sim(vectors_ndarray[i], vectors_ndarray[j])
                vector_similarities.append(new_sim)
                # Progress.
                count += 1
                progress_bar(count, total)
    else:
        # Create list with the similarity between the vectors.
        vector_similarities = [
            cosine_sim(vectors_ndarray[i], vectors_ndarray[j])
            for i in range(1, len(vectors_ndarray))
            for j in range(i)
        ]
    # Calculate the Average.
    average_sim = sum(vector_similarities) / len(vector_similarities)
    if show_progress:
        count += 1
        progress_bar(count, total)
    # Average of the Similarities.
    return average_sim


def mean_similarity_parallel(vectors_ndarray: list, show_progress=False):
    """
    Parallelism version of the mean_similarity() method.

    Find the average similarity between the vectors inside the given vector
    cluster/group 'vectors_ndarray'.

    Args:
        vectors_ndarray: List[Numpy.ndarray] with the vector inside the cluster.
        show_progress: A Bool representing whether we show the progress of
            the method or not.
    Returns:
        Float between 0.0 and 1.0 with the average similarity between the
            vectors.
    """
    # Determine the number of cores to be used. (I made my own formula)
    total_iterations = (len(vectors_ndarray) * (len(vectors_ndarray) - 1)) // 2
    optimal_cores = min(multiprocessing.cpu_count(), MAX_CORES)
    efficiency_mult = min(float(1), total_iterations / PEAK_ITERATIONS)
    core_count = max(2, round(efficiency_mult * optimal_cores))
    # Create chunk size to process the tasks in the cores.
    chunk_size = max(1, total_iterations // 100)

    # Create Tuple Parameters.
    tuple_params = [
        (vectors_ndarray[i], vectors_ndarray[j])
        for i in range(1, len(vectors_ndarray))
        for j in range(i)
    ]
    # Progress Variables.
    # Add the action of calculating the average to total.
    count = 0
    total = total_iterations + 1

    # Calculate the Similarities between the vectors using Parallelism.
    with multiprocessing.Pool(processes=core_count) as pool:
        # Use Pool.imap() if we have to show the progress.
        if show_progress:
            # Report Parallelization.
            progress_msg(f"Using Parallelization <{core_count} cores>")
            # Create Lazy iterator of results using Pool.imap()
            imap_results_iter = pool.imap(
                _custom_cosine_sim, tuple_params, chunksize=chunk_size
            )
            # Add the Similarities between the vectors to the list.
            vector_similarities = []
            for new_sim in imap_results_iter:
                # Add Similarity to list.
                vector_similarities.append(new_sim)
                # Progress.
                count += 1
                progress_bar(count, total)
        # No need to show progress, use Pool.map()
        else:
            # Create List with the similarities.
            vector_similarities = pool.map(
                _custom_cosine_sim, tuple_params, chunksize=chunk_size
            )
    # Calculate the Average.
    average_sim = sum(vector_similarities) / len(vector_similarities)
    if show_progress:
        count += 1
        progress_bar(count, total)
    # Average of the Similarities.
    return average_sim


def _custom_cosine_sim(vectors_tuple: tuple):
    """
    Custom-made version of the method cosine_sim() to use with parallelization.
    """
    embed_a, embed_b = vectors_tuple
    return cosine_sim(embed_a, embed_b)


if __name__ == '__main__':
    # Record Program Runtime.
    _stopwatch = TimeKeeper()
    # Terminal Arguments.
    _args = sys.argv

    # # Create a list of tuples to get the Top N elements.
    # _origin = random.sample(range(12, 100), 20)
    # _tuples = [(n, n) for n in _origin]
    # print("\nElements to Sort:")
    # pprint(_tuples)
    # # --------------------------------------------------
    # # Show Top Elements
    # _top_num = 10
    # print(f"\nTop {_top_num} elements:")
    # _tops = find_top_n(_tuples, n=_top_num, top_max=True)
    # pprint(_tops)

    # Find the Average Similarities inside a group of vector.
    _vectors = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.0, 2.0, 3.0, 4.0]),
        np.array([3.0, 4.0, 3.0, 4.0]),
        np.array([4.0, 5.0, 6.0, 7.0]),
        np.array([9.0, 8.0, 7.0, 6.0]),
        np.array([5.0, 9.0, 2.0, 1.0]),
    ]
    _average_sim = mean_similarity(vectors_ndarray=_vectors)
    print(f"\nMean Similarity of the Vectors: {_average_sim}")

    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
