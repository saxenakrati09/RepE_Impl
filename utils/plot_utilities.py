
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.common_utilities import print_colored_terminal

def plot_lat_scans(input_ids, rep_reader_scores_dict, layer_slice, filename, seed=1):
    """
    Visualizes the neural activity of a model for specific input tokens and layer slice.

    Parameters
    ----------
    input_ids : list of str
        A list of input tokens for the model.
    rep_reader_scores_dict : dict
        A dictionary where keys are representation names and values are arrays of neural activity scores.
    layer_slice : slice
        A slice object specifying the range of layers to visualize.
    filename : str
        The name of the file to save the plot.
    seed : int, optional
        A random seed for reproducibility (default is None).

    Returns
    -------
    None
        This function generates a heatmap plot of the neural activity and displays it.
    """

    # Set the random seed if provided
    np.random.seed(seed)

    # Iterate over each representation and its corresponding scores
    for rep, scores in rep_reader_scores_dict.items():
        # Find the starting token index that begins with 'ĠA', default to 0 if not found
        start_tok = next((i for i, tok in enumerate(input_ids) if tok.startswith('ĠA')), 0)

        # Extract and standardize scores for the specified token range and layer slice
        standardized_scores = np.array(scores)[start_tok:start_tok+50, layer_slice]

        # Define a bound for clipping the scores based on mean and standard deviation
        bound = np.mean(standardized_scores) + np.std(standardized_scores)
        bound = 2.3  # Override the bound with a fixed value

        # Set scores below a threshold to 1 and clip scores to the defined bound
        threshold = 0
        standardized_scores[np.abs(standardized_scores) < threshold] = 1
        standardized_scores = standardized_scores.clip(-bound, bound)

        # Define the colormap for the heatmap
        cmap = 'coolwarm'

        # Create a new figure for the heatmap
        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)

        # Plot the heatmap of the standardized scores (transposed for layers on y-axis)
        sns.heatmap(-standardized_scores.T, cmap=cmap, linewidth=0.5, annot=False, fmt=".3f", vmin=-bound, vmax=bound)

        # Configure y-axis ticks and labels
        ax.tick_params(axis='y', rotation=0)

        # Set x-axis and y-axis labels
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")

        # Set x-axis ticks to appear every 5 tokens
        ax.set_xticks(np.arange(0, len(standardized_scores), 5)[1:])
        ax.set_xticklabels(np.arange(0, len(standardized_scores), 5)[1:])

        ax.tick_params(axis='x', rotation=0)

        # Set y-axis ticks 
        ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
        ax.set_yticklabels(np.arange(0, len(standardized_scores[0]), 5)[::-1][1:])

        # Set the title of the heatmap
        ax.set_title("LAT Neural Activity")

    # Display the heatmap
    plt.show()



def plot_detection_results(input_ids, rep_reader_scores_dict, THRESHOLD, seed=1):
    """
    Plot the neural activity of a model on a given input.

    Parameters
    ----------
    input_ids : list of str
        The input tokens to the model.
    rep_reader_scores_dict : dict of str to array
        A dictionary mapping the names of the concepts to their corresponding
        neural activity.
    THRESHOLD : float
        The threshold value for the neural activity to be considered a concept.
    start_answer_token : str, optional
        The token that marks the start of the answer in the input, by default ":"
    seed : int, optional
        A random seed for reproducibility (default is 1).

    Returns
    -------
    None
    """

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Process input tokens to remove special characters for better visualization
    words = [token.replace('Ġ', '').replace('Ċ', ' ') for token in input_ids]

    # Initialize variables for iteration and word width
    y_pad = 0.3
    word_width = 0
    iter = 0

    # Define the concepts and styles for visualization
    selected_concepts = ["risk"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    # Iterate over each concept and its corresponding style
    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):
        # Retrieve scores for the current concept
        rep_scores = np.array(rep_reader_scores_dict[rep])

        # Remove outliers by capping extreme values
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean + 5 * std) | (rep_scores < mean - 5 * std)] = mean

        # Define the magnitude for normalization
        mag = max(0.3, np.abs(rep_scores).std() / 10)

        min_val, max_val = -mag, mag

        # Normalize scores based on the specified style
        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD  # Subtract threshold
            rep_scores = rep_scores / np.std(rep_scores[5:])  # Normalize by standard deviation
            min_val = np.min(rep_scores)
            max_val = np.max(rep_scores)
            if max_val - min_val != 0:
                rep_scores = 2 * (rep_scores - min_val) / (max_val - min_val) - 1
            else:
                rep_scores = np.zeros_like(rep_scores)

        if "flip" in n_style:
            rep_scores = -rep_scores  # Flip the scores if specified

        # Set scores below a certain threshold to zero
        rep_scores[np.abs(rep_scores) < 0.0] = 0

        # Apply selection style to filter scores
        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)  # Keep only negative scores
            rep_scores[rep_scores == 0] = mag  # Replace zero scores with the magnitude
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)  # Keep only positive scores
        print_colored_terminal(words, rep_scores)