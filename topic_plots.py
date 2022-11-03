# Gelin Eguinosa Rosique
# 2022

import matplotlib.pyplot as plt

from topic_utils import pwi_plot_data, homogeneity_plot_data, embeds_2d_data


# Vector Space Names.
space_names = {
    'mono_20': 'SBERT',
    'mix_20': 'SPECTER-SBERT',
    'test_20': 'Test Model',
}
# Colors Info.
empty_embed_color = 'gainsboro'
distinct_colors = {
    'red': '#e6194B',
    'green': '#3cb44b',
    'yellow': '#ffe119',
    'blue': '#4363d8',
    'orange': '#f58231',
    'purple': '#911eb4',
    'cyan': '#42d4f4',
    'magenta': '#f032e6',
    'lime': '#bfef45',
    'pink': '#fabed4',
    'teal': '#469990',
    'lavender': '#dcbeff',
    'brown': '#9A6324',
    'beige': '#fffac8',
    'maroon': '#800000',
    'mint': '#aaffc3',
    'olive': '#808000',
    'apricot': '#ffd8b1',
    'navy': '#000075',
    'grey': '#a9a9a9',
    'white': '#ffffff',
    'black': '#000000',
}
accessibility_order = [
    'white', 'black', 'grey', 'blue', 'yellow', 'navy', 'maroon', 'lavender',
    'orange', 'mint', 'beige', 'brown', 'teal', 'pink', 'magenta', 'cyan',
    'green', 'red', 'apricot', 'olive', 'lime', 'purple',
]
# Topic colors.
all_topics_colors = {
    'Topic_01': 'grey',
    'Topic_02': 'blue',
    'Topic_03': 'yellow',
    'Topic_04': 'navy',
    'Topic_05': 'maroon',
    'Topic_06': 'lavender',
    'Topic_07': 'orange',
    'Topic_08': 'mint',
    'Topic_09': 'beige',
    'Topic_10': 'brown',
    'Topic_11': 'teal',
    'Topic_12': 'pink',
    'Topic_13': 'magenta',
    'Topic_14': 'cyan',
    'Topic_15': 'green',
    'Topic_16': 'red',
    'Topic_17': 'apricot',
    'Topic_18': 'olive',
    'Topic_19': 'lime',
    'Topic_20': 'purple',
}
solo_topics_colors = {
    'Topic_01': 'black',
    'Topic_02': 'blue',
    'Topic_03': 'yellow',
    'Topic_04': 'navy',
    'Topic_05': 'maroon',
    'Topic_06': 'lavender',
    'Topic_07': 'orange',
    'Topic_08': 'mint',
    'Topic_09': 'beige',
    'Topic_10': 'brown',
    'Topic_11': 'teal',
    'Topic_12': 'pink',
    'Topic_13': 'magenta',
    'Topic_14': 'cyan',
    'Topic_15': 'green',
    'Topic_16': 'red',
    'Topic_17': 'apricot',
    'Topic_18': 'olive',
    'Topic_19': 'lime',
    'Topic_20': 'purple',
}


def generate_homogeneity_plot(homog_type: str, save_plot=False, f_format='pdf'):
    """
    Show or Save a Homogeneity Plot comparing the Mono and Mix Topic Models.
    """
    # Get the Values to use on the Plot.
    x_values, mono_values, mix_values = homogeneity_plot_data(f'homog_{homog_type}')
    # Create Plot.
    fig, ax = plt.subplots()
    ax.plot(x_values, mono_values, label='Sentence-BERT')
    ax.plot(x_values, mix_values, label='SPECTER-SBERT')
    ax.set_xlabel("Número de Tópicos")
    ax.set_ylabel(f"Homogeneidad {homog_type.title()}")
    ax.set_title(
        f"Comparación de Modelos de Tópicos\ncon Homogeneidad {homog_type.title()}"
    )
    # Set the Range of the Axes & Legend.
    # ax.axis([0, 100, 0, max(max(mono_values), max(mix_values))])
    ax.axis([0, 100, 0, 85])
    ax.legend()
    # Show or Save Model.
    if save_plot:
        plt.savefig(
            f"temp_data/comparison_homogeneity_{homog_type}.{f_format}",
            format=f_format
        )
    else:
        plt.show()


def generate_pwi_plot(pwi_type: str, save_plot=False, f_format='pdf', other_models=False):
    """
    Show or Save a PWI Plot comparing the Mono and Mix Topic Models. It can also
    include data from the LDA and Doc2Vec Models (50 words).
    """
    # - Get the Values to use on the Plot -
    x_values, mono_values, mix_values = pwi_plot_data(pwi_type)
    # - Create Plot -
    fig, ax = plt.subplots()
    ax.plot(x_values, mono_values, label='Sentence-BERT')
    ax.plot(x_values, mix_values, label='SPECTER-SBERT')
    # - See if we have to include the LDA & Doc2Vec Values -
    if other_models and '50' in pwi_type:
        # Doc2Vec Values.
        doc2vec_tfidf = [0.0, 10.1, 14.4, 18.8, 21.8, 24.5, 25.2, 26.1, 27.6, 28.9, 30.7]
        doc2vec_exact = [0.0, 11.5, 16.4, 21.4, 24.8, 27.9, 28.8, 29.7, 31.5, 32.9, 34.9]
        # LDA Values.
        lda_tfidf = [0.0, 2.5, 3.2, 3.6, 4.5, 5.3, 6.5, 7.1, 7.2, 7.8, 9.0]
        lda_exact = [0.0, 2.9, 3.7, 4.1, 5.1, 6.0, 7.4, 8.0, 8.2, 8.9, 10.3]
        # Select Data.
        if 'tf-idf' in pwi_type:
            doc2vec_values = doc2vec_tfidf
            lda_values = lda_tfidf
        elif 'exact' in pwi_type:
            doc2vec_values = doc2vec_exact
            lda_values = lda_exact
        else:
            raise NameError(f"Doc2Vec doesn't have data for {pwi_type}")
        # Plot Doc2Vec & LDA.
        ax.plot(x_values, doc2vec_values, label='Doc2Vec')
        ax.plot(x_values, lda_values, label='LDA')
    # - Create Plot Names & Labels -
    ax.set_xlabel("Número de Tópicos")
    ax.set_ylabel("Ganancia de Información")
    pwi_name = 'PWI-exact' if 'exact' in pwi_type else 'PWI-tf-idf'
    num_words = pwi_type.split('_')[-1]
    ax.set_title(
        f"Comparación de Modelos de Tópicos\ncon {pwi_name} ({num_words} palabras)"
    )
    # - Set the Range of the Axes & Legend -
    # ax.axis([0, 100, 0, max(max(mono_values), max(mix_values))])
    ax.axis([0, 100, 0, 110])
    ax.legend()
    # - Show or Save Model -
    if save_plot:
        plt.savefig(
            f"temp_data/comparison_pwi_{pwi_type}_words.{f_format}",
            format=f_format
        )
    else:
        plt.show()


def generate_vector_space(data_name: str, save_plot=False, f_format='pdf'):
    """
    Create an image of the Vector Space of a Topic Model.
    """
    # Get Dictionary with the 2D Info of the Topic Model.
    model_2d_info = embeds_2d_data(data_name)
    # Create Figure & Axes.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Rest of Topic IDs.
    for topic_id in model_2d_info.keys():
        # Extract the 2D Info.
        topic_info = model_2d_info[topic_id]
        topic_2d_embed = topic_info['topic_embed']
        doc_2d_embeds = topic_info['doc_embeds']
        # topic_top_words = topic_info['top_words']

        # Topic Values.
        topic_x_value = topic_2d_embed[0]
        topic_y_value = topic_2d_embed[1]
        # Document Values.
        doc_x_values = []
        doc_y_values = []
        for x, y in doc_2d_embeds.values():
            doc_x_values.append(x)
            doc_y_values.append(y)

        # Plot the Vectors in the Topic 'topic_id'.
        topic_color = solo_topics_colors[topic_id]
        color_code = distinct_colors[topic_color]
        ax.scatter(
            doc_x_values, doc_y_values,
            color=color_code, s=0.1
        )
        # Plot the Text Name.
        plt.text(
            topic_x_value, topic_y_value, s=topic_id,
            fontdict=dict(color='black', size=6),
            bbox=dict(facecolor='white', alpha=0.65)
        )

    # Name the Plot.
    space_name = space_names[data_name]
    ax.set_title(f"Vector Space {space_name}")
    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    # Show or Save the Model.
    plt.tight_layout()
    if save_plot:
        plt.savefig(
            f"temp_data/vector_space_{data_name}.{f_format}",
            format=f_format
        )
    else:
        plt.show()


def generate_topic_space(
        data_name: str, main_topic: str, save_plot=False, f_format='pdf'
):
    """
    Create an image of Vector Space from the Topic Model in 'data_name'
    highlighting the given Topic 'main_topic'.
    """
    # Get Dictionary with the 2D Info of the Topic Model.
    model_2d_info = embeds_2d_data(data_name)
    # Create Figure and Axes.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Iterate through the data of the Topics and plot them.
    for topic_id in model_2d_info.keys():
        # Extract the 2D Info.
        topic_info = model_2d_info[topic_id]
        topic_2d_embed = topic_info['topic_embed']
        doc_2d_embeds = topic_info['doc_embeds']
        # topic_top_words = topic_info['top_words']

        # Topic Values.
        topic_x_value = topic_2d_embed[0]
        topic_y_value = topic_2d_embed[1]
        # Document Values.
        doc_x_values = []
        doc_y_values = []
        for x, y in doc_2d_embeds.values():
            doc_x_values.append(x)
            doc_y_values.append(y)

        # Plot the Documents.
        if topic_id == main_topic:
            # Plot the Highlighted Topic.
            topic_color = solo_topics_colors[topic_id]
            color_code = distinct_colors[topic_color]
            ax.scatter(doc_x_values, doc_y_values, color=color_code, s=0.1)
            # Add Label for the Highlighted Topic.
            plt.text(
                topic_x_value, topic_y_value, s=topic_id,
                fontdict=dict(color='black', size=6),
                bbox=dict(facecolor='white', alpha=0.65)
            )
        else:
            # Plot the Background Topics.
            ax.scatter(
                doc_x_values, doc_y_values, color=empty_embed_color, s=0.1
            )

    # Name the Plot.
    space_name = space_names[data_name]
    ax.set_title(f"{main_topic} in the Vector Space {space_name}")
    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    # Show or Save the Model.
    plt.tight_layout()
    if save_plot:
        plt.savefig(
            f"temp_data/vector_space_{data_name}_{main_topic}.{f_format}",
            format=f_format
        )
    else:
        plt.show()


if __name__ == '__main__':
    # # Create PWI Plot.
    # print("\nCreating PWI Plot...")
    # generate_pwi_plot(pwi_type='tf-idf_50', save_plot=False, other_models=True)
    # # generate_pwi_plot(pwi_type='exact_50', save_plot=False, other_models=True)

    # # Create Homogeneity Plot.
    # print("\nCreating Homogeneity Plot...")
    # _f_format = 'png'
    # generate_homogeneity_plot(
    #     homog_type='topic-doc', save_plot=True, f_format=_f_format
    # )
    # generate_homogeneity_plot(
    #     homog_type='doc-doc', save_plot=True, f_format=_f_format
    # )

    # Create Vector Space Image.
    _data_name = 'mono_20'
    _save = True
    _f_format = 'png'
    # -----------------------------------------------------
    # Plot Vector Space.
    print(f"\nCreating Vector Space for Topic Model {_data_name}")
    generate_vector_space(data_name=_data_name, save_plot=_save, f_format=_f_format)
    # -----------------------------------------------------
    # # Create Topic Image in Vector Space.
    # _topic_id = 'Topic_04'
    # print(f"\nCreating Topic Vector Space for <{_topic_id}> in data <{_data_name}>...")
    # generate_topic_space(
    #     data_name=_data_name, main_topic=_topic_id,
    #     save_plot=_save, f_format=_f_format
    # )

    print("\nDone.\n")
