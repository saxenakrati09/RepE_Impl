from transformers import PreTrainedTokenizer
import random
import pandas as pd
import numpy as np



def risk_function_dataset(data_path: str, 
                          tokenizer: PreTrainedTokenizer, 
                          user_tag: str = "", 
                          assistant_tag: str = "", 
                          closing_tag: str = "", 
                          num_train = 512, 
                          seed: int = 0) -> tuple[list, list]:
    """
    Processes data to create training and testing datasets based on risk.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data from the CSV file
    df = pd.read_csv(data_path)
    #shuffle the data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Extract true statements (label = 1) and false statements (label = 0)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    # Initialize lists to store risk and nonrisk prompts
    risk_statements = []
    nonrisk_statements = []

    # Process each true statement
    for statement in true_statements:
        # Tokenize the statement
        tokens = tokenizer.tokenize(statement)
        
        # Iterate through tokens to create truncated versions of the statement
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]  # Truncate tokens up to the current index
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)  # Convert tokens back to string

            # Create a risk prompt using the truncated statement
            # risk_prompt = (
            #     f"{user_tag}Pretend you're a financial advisor making risky investment recommendations.\n"
            #     f"{assistant_tag}{truncated_statement}\n{closing_tag}"
            # )
            # # Create a non-risk prompt using the truncated statement
            # non_risk_prompt = (
            #     f"{user_tag}Pretend you're a financial advisor making safe investment recommendations.\n"
            #     f"{assistant_tag}{truncated_statement}\n{closing_tag}"
            # )

            risk_prompt = (
                f"{user_tag}Based on the financial scenario below, suggest an investment strategy that embraces a high-risk approach aiming for higher returns.\n"
                f"{assistant_tag}{truncated_statement}\n{closing_tag}"
            )
            # Create a non-risk prompt using the truncated statement
            non_risk_prompt = (
                f"{user_tag}Based on the financial scenario below, suggest an investment strategy that prioritizes low-risk and stable returns.\n"
                f"{assistant_tag}{truncated_statement}\n{closing_tag}"
            )
            
            # Append the prompts to their respective lists
            risk_statements.append(risk_prompt)
            nonrisk_statements.append(non_risk_prompt)


    # Combine risk and nonrisk prompts into pairs
    combined_data = [[risk, nonrisk] for risk, nonrisk in zip(risk_statements, nonrisk_statements)]

    # Select the first `num_train` pairs for training data
    train_data = combined_data[:num_train]

    # Initialize a list to store training labels
    train_labels = []
    for d in train_data:
        true_s = d[0]  # The risk statement is the true label
        random.shuffle(d)  # Shuffle the pair to randomize order
        train_labels.append([s == true_s for s in d])  # Create a binary label indicating which statement is true
    
    # Flatten the training data into a single list
    train_data = np.concatenate(train_data).tolist()

    # Create test data by reshaping risk and nonrisk prompts into pairs
    reshaped_data = np.array([[risk, nonrisk] for risk, nonrisk in zip(risk_statements[:-1], nonrisk_statements[1:])]).flatten()

    # Select test data from the reshaped data, excluding the training samples
    test_data = reshaped_data[num_train:num_train*2].tolist()

    # Log the size of the training and testing datasets
    # ic(f"Train data: {len(train_data)}")
    # ic(f"Test data: {len(test_data)}")

    # Return the training and testing datasets along with their labels
    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }