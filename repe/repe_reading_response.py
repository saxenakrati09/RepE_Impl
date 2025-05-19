from typing import List, Union, Optional, Dict
from transformers import Pipeline
import torch
import numpy as np
from icecream import ic
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from itertools import islice
import torch

def project_onto_direction(H, direction, device="cuda"):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    # Calculate the magnitude of the direction vector
     # Ensure H and direction are on the same device (CPU or GPU)
    if not isinstance(direction, torch.Tensor):
        H = torch.Tensor(H).to(device)
    if type(direction) != torch.Tensor:
        direction = torch.Tensor(direction)
        direction = direction.to(H.device)
    mag = torch.norm(direction)
    assert not torch.isinf(mag).any()
    # Calculate the projection
    projection = H.matmul(direction) / mag
    return projection

def recenter(x, mean=None):
    # Convert the input x to a PyTorch tensor and move it to the GPU (CUDA device).
    x = torch.Tensor(x).cuda()
    
    if mean is None:
        # If no mean is provided, calculate the mean of x along the first axis (batch dimension),
        # keeping the dimensions for broadcasting, and move it to the GPU.
        mean = torch.mean(x, axis=0, keepdims=True).cuda()
    else:
        # If a mean is provided, convert it to a PyTorch tensor and move it to the GPU.
        mean = torch.Tensor(mean).cuda()
    
    # Subtract the mean from x to recenter it around zero.
    return x - mean

class RepReader(ABC):
    """Class to identify and store concept directions.
    
    Subclasses implement the abstract methods to identify concept directions 
    for each hidden layer via strategies including PCA, embedding vectors 
    (aka the logits method), and cluster means.

    RepReader instances are used by RepReaderPipeline to get concept scores.

    Directions can be used for downstream interventions."""

    @abstractmethod
    def __init__(self) -> None:
        self.direction_method = None
        self.directions = None # directions accessible via directions[layer][component_index]
        self.direction_signs = None # direction of high concept scores (mapping min/max to high/low)

    @abstractmethod
    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        """Get concept directions for each hidden layer of the model
        
        Args:
            model: Model to get directions for
            tokenizer: Tokenizer to use
            hidden_states: Hidden states of the model on the training data (per layer)
            hidden_layers: Layers to consider

        Returns:
            directions: A dict mapping layers to direction arrays (n_components, hidden_size)
        """
        pass 

    def get_signs(self, hidden_states, train_choices, hidden_layers, device="cuda"):
        """Given labels for the training data hidden_states, determine whether the
        negative or positive direction corresponds to low/high concept 
        (and return corresponding signs -1 or 1 for each layer and component index)
        
        NOTE: This method assumes that there are 2 entries in hidden_states per label, 
        aka len(hidden_states[layer]) == 2 * len(train_choices). For example, if 
        n_difference=1, then hidden_states here should be the raw hidden states
        rather than the relative (i.e. the differences between pairs of examples).

        Args:
            hidden_states: Hidden states of the model on the training data (per layer)
            train_choices: Labels for the training data
            hidden_layers: Layers to consider

        Returns:
            signs: A dict mapping layers to sign arrays (n_components,)
        """        
        signs = {}

        if self.needs_hiddens and hidden_states is not None and len(hidden_states) > 0:
            for layer in hidden_layers:    
                assert hidden_states[layer].shape[0] == 2 * len(train_choices), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(train_choices)})"
                
                signs[layer] = []
                for component_index in range(self.n_components):
                    transformed_hidden_states = project_onto_direction(hidden_states[layer], self.directions[layer][component_index], device)
                    projected_scores = [transformed_hidden_states[i:i+2] for i in range(0, len(transformed_hidden_states), 2)]

                    outputs_min = [1 if min(o) == o[label] else 0 for o, label in zip(projected_scores, train_choices)]
                    outputs_max = [1 if max(o) == o[label] else 0 for o, label in zip(projected_scores, train_choices)]
                    
                    signs[layer].append(-1 if np.mean(outputs_min) > np.mean(outputs_max) else 1)
        else:
            for layer in hidden_layers:    
                signs[layer] = [1 for _ in range(self.n_components)]

        return signs


    def transform(self, hidden_states, hidden_layers, component_index, device):
        """Project the hidden states onto the concept directions in self.directions

        Args:
            hidden_states: dictionary with entries of dimension (n_examples, hidden_size)
            hidden_layers: list of layers to consider
            component_index: index of the component to use from self.directions
            device: the device (e.g., 'cuda' or 'cpu') to perform computations on

        Returns:
            transformed_hidden_states: dictionary with entries of dimension (n_examples,)
        """

        # Ensure the component index is valid and within the number of components
        assert component_index < self.n_components

        # Initialize a dictionary to store the transformed hidden states for each layer
        transformed_hidden_states = {}

        # Iterate over the specified hidden layers
        for layer in hidden_layers:
            
            # Get the hidden states for the current layer
            layer_hidden_states = hidden_states[layer]

            # If the class has stored training means, recenter the hidden states using the mean
            if hasattr(self, 'H_train_means'):
                layer_hidden_states = recenter(layer_hidden_states, mean=self.H_train_means[layer])

            # Project the hidden states onto the specified concept direction (e.g., PCA component)
            H_transformed = project_onto_direction(layer_hidden_states, self.directions[layer][component_index], device)

            # Store the transformed hidden states in the dictionary, converting them to numpy arrays
            transformed_hidden_states[layer] = H_transformed.cpu().numpy()

        # Return the dictionary containing the transformed hidden states for all layers
        return transformed_hidden_states

class PCARepReader(RepReader):
    """Extract directions via PCA"""
    needs_hiddens = True  # Indicates that this class requires hidden states to compute directions.

    def __init__(self, n_components=1):
        super().__init__()  # Call the parent class constructor.
        self.n_components = n_components  # Number of PCA components to extract.
        self.H_train_means = {}  # Dictionary to store the mean of training hidden states for each layer.

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        """Get PCA components for each layer"""
        directions = {}  # Dictionary to store PCA directions for each layer.

        for layer in hidden_layers:  # Iterate over the specified hidden layers.
            H_train = hidden_states[layer]  # Get the hidden states for the current layer.
            H_train_mean = H_train.mean(axis=0, keepdims=True)  # Compute the mean of the hidden states.
            self.H_train_means[layer] = H_train_mean  # Store the mean for later use.
            H_train = recenter(H_train, mean=H_train_mean).cpu()  # Recenter the hidden states and move to CPU.
            H_train = np.vstack(H_train)  # Convert the hidden states to a 2D NumPy array.
            pca_model = PCA(n_components=self.n_components, whiten=False).fit(H_train)  # Fit a PCA model.

            directions[layer] = pca_model.components_  # Store the PCA components for the current layer.
            self.n_components = pca_model.n_components_  # Update the number of components based on the PCA model.
        
        return directions  # Return the computed directions.

    def get_signs(self, hidden_states, train_labels, hidden_layers):
        """Determine the signs of the PCA components for each layer"""
        signs = {}  # Dictionary to store the signs for each layer.

        for layer in hidden_layers:  # Iterate over the specified hidden layers.
            # Ensure the number of hidden states matches the number of training labels.
            assert hidden_states[layer].shape[0] == len(np.concatenate(train_labels)), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(np.concatenate(train_labels))})"
            layer_hidden_states = hidden_states[layer]  # Get the hidden states for the current layer.

            # Recenter the hidden states using the stored mean for the current layer.
            layer_hidden_states = recenter(layer_hidden_states, mean=self.H_train_means[layer])

            # Initialize an array to store the signs for each PCA component.
            layer_signs = np.zeros(self.n_components)
            for component_index in range(self.n_components):  # Iterate over the PCA components.

                # Project the hidden states onto the current PCA component.
                transformed_hidden_states = project_onto_direction(layer_hidden_states, self.directions[layer][component_index]).cpu()
                
                # Group the transformed hidden states by training label.
                pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in train_labels[:i]), sum(len(c) for c in train_labels[:i+1]))) for i in range(len(train_labels))]

                # Compute the proportion of cases where the minimum value corresponds to the correct label.
                pca_outputs_min = np.mean([o[train_labels[i].index(1)] == min(o) for i, o in enumerate(pca_outputs_comp)])
                # Compute the proportion of cases where the maximum value corresponds to the correct label.
                pca_outputs_max = np.mean([o[train_labels[i].index(1)] == max(o) for i, o in enumerate(pca_outputs_comp)])

                # Determine the sign of the component based on the difference between max and min proportions.
                layer_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min))
                if layer_signs[component_index] == 0:  # If there's a tie, default to a positive sign.
                    layer_signs[component_index] = 1

            signs[layer] = layer_signs  # Store the signs for the current layer.

        return signs  # Return the computed signs.

        
class ClusterMeanRepReader(RepReader):
    """Get the direction that is the difference between the mean of the positive and negative clusters."""
    n_components = 1
    needs_hiddens = True

    def __init__(self):
        super().__init__()

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):

        # train labels is necessary to differentiate between different classes
        train_choices = kwargs['train_choices'] if 'train_choices' in kwargs else None
        assert train_choices is not None, "ClusterMeanRepReader requires train_choices to differentiate two clusters"
        for layer in hidden_layers:
            assert len(train_choices) == len(hidden_states[layer]), f"Shape mismatch between hidden states ({len(hidden_states[layer])}) and labels ({len(train_choices)})"

        train_choices = np.array(train_choices)
        neg_class = np.where(train_choices == 0)
        pos_class = np.where(train_choices == 1)

        directions = {}
        for layer in hidden_layers:
            H_train = np.array(hidden_states[layer])

            H_pos_mean = H_train[pos_class].mean(axis=0, keepdims=True)
            H_neg_mean = H_train[neg_class].mean(axis=0, keepdims=True)

            directions[layer] = H_pos_mean - H_neg_mean
        
        return directions


class RandomRepReader(RepReader):
    """Get random directions for each hidden layer. Do not use hidden 
    states or train labels of any kind."""

    def __init__(self, needs_hiddens=True):
        super().__init__()

        self.n_components = 1
        self.needs_hiddens = needs_hiddens

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):

        directions = {}
        for layer in hidden_layers:
            directions[layer] = np.expand_dims(np.random.randn(model.config.hidden_size), 0)

        return directions


DIRECTION_FINDERS = {
    'pca': PCARepReader,
    'cluster_mean': ClusterMeanRepReader,
    'random': RandomRepReader,
}

class RepReadingPipeline(Pipeline):
    """
    A pipeline for extracting and transforming hidden states from a model using various representation readers (RepReaders).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_hidden_states(
            self, 
            outputs,
            rep_token: Union[str, int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        """
        Extract hidden states for specified layers and tokens from model outputs.

        Args:
        - outputs: Model outputs containing hidden states.
        - rep_token: Token index to extract hidden states for (default is -1, the last token).
        - hidden_layers: List of layer indices to extract hidden states from (default is -1, the last layer).
        - which_hidden_states: Specifies which hidden states to use (e.g., 'encoder' or 'decoder').

        Returns:
        - dict: A dictionary mapping layer indices to extracted hidden states.
        """
        # If the model has both encoder and decoder hidden states, select the appropriate one.
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']

        # Extract the predicted output IDs from the logits.
        output_ids = outputs['logits'].argmax(dim=-1).tolist()

        # Initialize a dictionary to store hidden states for each layer.
        response_hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            # Convert bfloat16 tensors to float for compatibility.
            if hidden_states.dtype == torch.bfloat16:
                hidden_states = hidden_states.float()

            response_hidden_states = []
            # Iterate over the batch to extract hidden states for the specified token.
            for i in range(len(output_ids)):
                response_hidden_states.append(hidden_states[i, rep_token, :].detach())

            # Stack the hidden states for the batch and store them in the dictionary.
            response_hidden_states_layers[layer] = torch.stack(response_hidden_states)

        return response_hidden_states_layers

    def _sanitize_parameters(self, 
                             rep_reader: RepReader=None,
                             rep_token: Union[str, int]=-1,
                             hidden_layers: Union[List[int], int]=-1,
                             component_index: int=0,
                             which_hidden_states: Optional[str]=None,
                             **tokenizer_kwargs):
        """
        Prepare and validate parameters for the pipeline.

        Args:
        - rep_reader: An instance of RepReader to transform hidden states.
        - rep_token: Token index to extract hidden states for.
        - hidden_layers: List of layer indices to extract hidden states from.
        - component_index: Index of the component to use from the RepReader's directions.
        - which_hidden_states: Specifies which hidden states to use (e.g., 'encoder' or 'decoder').
        - **tokenizer_kwargs: Additional arguments for the tokenizer.

        Returns:
        - tuple: Preprocessing, forward, and postprocessing parameters.
        """
        preprocess_params = tokenizer_kwargs
        forward_params = {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        # Ensure hidden_layers is a list.
        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]

        # Validate that the number of directions matches the number of hidden layers.
        assert rep_reader is None or len(rep_reader.directions) == len(hidden_layers), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"
        forward_params['rep_reader'] = rep_reader
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states

        return preprocess_params, forward_params, postprocess_params

    def preprocess(
            self, 
            inputs: Union[str, List[str], List[List[str]]],
            **tokenizer_kwargs):
        """
        Preprocess input data using the tokenizer or image processor.

        Args:
        - inputs: Input data to preprocess.
        - **tokenizer_kwargs: Additional arguments for the tokenizer.

        Returns:
        - dict: Tokenized or processed input data.
        """
        if self.image_processor:
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):
        """
        Postprocess the outputs (no additional processing in this implementation).

        Args:
        - outputs: Model outputs.

        Returns:
        - outputs: Unmodified model outputs.
        """
        return outputs

    def _forward(self, model_inputs, rep_token, hidden_layers, rep_reader=None, component_index=0, which_hidden_states=None, pad_token_id=None, device="cuda"):
        """
        Perform a forward pass through the model to extract hidden states and optionally transform them using a RepReader.

        Args:
        - model_inputs: Input data for the model, including tokenized input IDs and attention masks.
        - rep_token: Token index to extract hidden states for.
        - hidden_layers: List of layer indices to extract hidden states from.
        - rep_reader: An instance of RepReader to transform the hidden states.
        - component_index: Index of the component to use from the RepReader's directions.
        - which_hidden_states: Specifies which hidden states to use (e.g., 'encoder' or 'decoder').
        - pad_token_id: ID of the padding token (not used in this implementation).
        - device: Device to perform computations on.

        Returns:
        - dict: Transformed hidden states if a RepReader is provided, otherwise raw hidden states.
        """
        with torch.no_grad():
            # Check if the model is an encoder-decoder model.
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                # Create a decoder input consisting of padding tokens for each input in the batch.
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input

            # Perform a forward pass through the model and request hidden states as part of the output.
            outputs = self.model(**model_inputs, output_hidden_states=True)

        # Extract hidden states for the specified layers and token from the model outputs.
        response_hidden_states_layers = self._get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)

        # If no RepReader is provided, return the raw hidden states.
        if rep_reader is None:
            return response_hidden_states_layers

        # Use the RepReader to transform the hidden states based on the specified component and device.
        transformed_response_hidden_states = rep_reader.transform(response_hidden_states_layers, hidden_layers, component_index, device)

        return transformed_response_hidden_states

    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args):
        """
        Processes a list of input strings to extract hidden states for specified layers.

        Args:
        - train_inputs: Input data which can be a string, list of strings, or list of list of strings.
        - rep_token: Token used as the representation.
        - hidden_layers: Specifies which layers' hidden states to retrieve.
        - batch_size: Number of samples to process in each batch.
        - which_hidden_states: Indicates which part of the model to obtain hidden states from ('encoder', 'decoder', or None).
        - **tokenizer_args: Additional arguments passed to the tokenizer.

        Returns:
        - dict: A dictionary with keys as layer indices and values as 2D numpy arrays of hidden states.
        """
        hidden_states_outputs = self(
            train_inputs, 
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            batch_size=batch_size,
            rep_reader=None,
            which_hidden_states=which_hidden_states,
            **tokenizer_args
        )

        hidden_states = {layer: [] for layer in hidden_layers}

        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])

        return {k: np.vstack(v) for k, v in hidden_states.items()}
    
    def _validate_params(self, n_difference, direction_method):
        """
        Validate parameters for the direction-finding method.

        Args:
        - n_difference: Number of differences to compute for relative hidden states.
        - direction_method: Method to find directions (e.g., PCA, cluster mean, random).
        """
        if direction_method == 'clustermean':
            assert n_difference == 1, "n_difference must be 1 for clustermean"

    def get_directions(
            self, 
            train_inputs: Union[str, List[str], List[List[str]]],
            rep_token: Union[str, int]=-1,
            hidden_layers: Union[str, int]=-1,
            n_difference: int = 1,
            batch_size: int = 8,
            train_labels: List[int] = None,
            direction_method: str = 'pca',
            direction_finder_kwargs: dict = {},
            which_hidden_states: Optional[str]=None,
            **tokenizer_args):
        """
        Train a RepReader on the training data to find concept directions.

        Args:
        - train_inputs: Input data for training.
        - rep_token: Token used as the representation.
        - hidden_layers: Specifies which layers' hidden states to retrieve.
        - n_difference: Number of differences to compute for relative hidden states.
        - batch_size: Batch size for processing the training data.
        - train_labels: Labels for the training data.
        - direction_method: Method to find directions (e.g., PCA, cluster mean, random).
        - direction_finder_kwargs: Additional arguments for the direction finder.
        - which_hidden_states: Specifies which hidden states to use (e.g., 'encoder' or 'decoder').
        - **tokenizer_args: Additional arguments for the tokenizer.

        Returns:
        - RepReader: A trained RepReader instance with computed directions.
        """
        if not isinstance(hidden_layers, list):
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]

        self._validate_params(n_difference, direction_method)

        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)

        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            hidden_states = self._batched_string_to_hiddens(
                train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args
            )
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]

        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels
        )
        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
                hidden_states, train_labels, hidden_layers
            )

        return direction_finder
