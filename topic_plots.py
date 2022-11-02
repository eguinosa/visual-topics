# Gelin Eguinosa Rosique
# 2022

import matplotlib.pyplot as plt

from topic_utils import pwi_plot_data, homogeneity_plot_data


def generate_homogeneity_plot(homog_type: str, save_plot=False):
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
        plt.savefig(f"temp_data/comparison_homogeneity_{homog_type}.pdf", format='pdf')
    else:
        plt.show()


def generate_pwi_plot(pwi_type: str, save_plot=False, other_models=False):
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
        plt.savefig(f'temp_data/comparison_pwi_{pwi_type}_words.pdf', format='pdf')
    else:
        plt.show()


if __name__ == '__main__':
    # Create PWI Plot.
    print("\nCreating PWI Plot...")
    generate_pwi_plot(pwi_type='tf-idf_50', save_plot=False, other_models=True)
    # generate_pwi_plot(pwi_type='exact_50', save_plot=False, other_models=True)

    # # Create Homogeneity Plot.
    # print("\nCreating Homogeneity Plot...")
    # generate_homogeneity_plot(homog_type='topic-doc', save_plot=True)
    # generate_homogeneity_plot(homog_type='doc-doc', save_plot=True)

    print("\nDone.\n")

