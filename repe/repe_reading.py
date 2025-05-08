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
    x = torch.Tensor(x).cuda()
    if mean is None:
        mean = torch.mean(x,axis=0,keepdims=True).cuda()
    else:
        mean = torch.Tensor(mean).cuda()
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

    def get_signs(self, hidden_states, train_choices, hidden_layers, device):
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
        ic(hidden_states)
        ic(hidden_layers)
        # Ensure the component index is valid and within the number of components
        assert component_index < self.n_components

        # Initialize a dictionary to store the transformed hidden states for each layer
        transformed_hidden_states = {}

        # Iterate over the specified hidden layers
        for layer in hidden_layers:
            
            # Get the hidden states for the current layer
            layer_hidden_states = hidden_states[layer]
            ic(layer_hidden_states)
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
    needs_hiddens = True 

    def __init__(self, n_components=1):
        super().__init__()
        self.n_components = n_components
        self.H_train_means = {}

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        """Get PCA components for each layer"""
        directions = {}

        for layer in hidden_layers:
            H_train = hidden_states[layer]
            H_train_mean = H_train.mean(axis=0, keepdims=True)
            self.H_train_means[layer] = H_train_mean
            H_train = recenter(H_train, mean=H_train_mean).cpu()
            H_train = np.vstack(H_train)
            pca_model = PCA(n_components=self.n_components, whiten=False).fit(H_train)

            directions[layer] = pca_model.components_ # shape (n_components, n_features)
            self.n_components = pca_model.n_components_
        
        return directions

    def get_signs(self, hidden_states, train_labels, hidden_layers):

        signs = {}

        for layer in hidden_layers:
            assert hidden_states[layer].shape[0] == len(np.concatenate(train_labels)), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(np.concatenate(train_labels))})"
            layer_hidden_states = hidden_states[layer]

            # NOTE: since scoring is ultimately comparative, the effect of this is moot
            layer_hidden_states = recenter(layer_hidden_states, mean=self.H_train_means[layer])

            # get the signs for each component
            layer_signs = np.zeros(self.n_components)
            for component_index in range(self.n_components):

                transformed_hidden_states = project_onto_direction(layer_hidden_states, self.directions[layer][component_index]).cpu()
                
                pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in train_labels[:i]), sum(len(c) for c in train_labels[:i+1]))) for i in range(len(train_labels))]

                # We do elements instead of argmin/max because sometimes we pad random choices in training
                pca_outputs_min = np.mean([o[train_labels[i].index(1)] == min(o) for i, o in enumerate(pca_outputs_comp)])
                pca_outputs_max = np.mean([o[train_labels[i].index(1)] == max(o) for i, o in enumerate(pca_outputs_comp)])

       
                layer_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min))
                if layer_signs[component_index] == 0:
                    layer_signs[component_index] = 1 # default to positive in case of tie

            signs[layer] = layer_signs

        return signs
    

        
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_hidden_states(
            self, 
            outputs,
            rep_token: Union[str, int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']
        # ic(outputs.keys()) # ['logits', 'past_key_values', 'hidden_states']
        output_ids = outputs['logits'].argmax(dim=-1).tolist()
        # ic(len(output_ids)) # [32]
        # ic(output_ids) # [36]
        reasoning_indices = []
        for x in output_ids:
            reasoning_indices.append(x.index(151667))
        # ic(reasoning_indices)
        # hidden_states_layers = {}
        reasoning_hidden_states_layers = {}
        response_hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            # ic(hidden_states.shape) # [32, 36, 1024]
            # ic(hidden_states.shape) # [32, 1024]
            if hidden_states.dtype == torch.bfloat16:
                hidden_states = hidden_states.float()
            reasoning_hidden_states = []
            response_hidden_states = []

            for i in range(len(output_ids)):  # Iterate over the batch
                reasoning_hidden_states.append(hidden_states[i, reasoning_indices[i] - 1, :].detach())  # Last token of reasoning
                response_hidden_states.append(hidden_states[i, -1, :].detach())  # Last token of response

            # Stack the reasoning and response hidden states for the batch
            reasoning_hidden_states_layers[layer] = torch.stack(reasoning_hidden_states)
            # ic(reasoning_hidden_states_layers[layer].shape) # [32, 1024]
            response_hidden_states_layers[layer] = torch.stack(response_hidden_states)
            # ic(response_hidden_states_layers[layer].shape) # [32, 1024]
            # hidden_states =  hidden_states[:, rep_token, :].detach()
            # ic(hidden_states.shape) # [32, 1024]
            # for i in range(len(output_ids)):
            #     reasoning_hidden_states = hidden_states[:reasoning_indices[i], :].detach()
            #     response_hidden_states = hidden_states[reasoning_indices[i]:, :].detach()
            #     ic(reasoning_hidden_states.shape, response_hidden_states.shape)
            #     reasoning_hidden_states_layers[layer] = reasoning_hidden_states.detach()
            #     response_hidden_states_layers[layer] = response_hidden_states.detach()
            # hidden_states_layers[layer] = hidden_states.detach()
            # reasoning_hidden_states_layers[layer] = reasoning_hidden_states.detach()
            # response_hidden_states_layers[layer] = response_hidden_states.detach()
            # ic(reasoning_hidden_states_layers)
            # ic(response_hidden_states_layers)
        return reasoning_hidden_states_layers, response_hidden_states_layers

    def _sanitize_parameters(self, 
                             rep_reader: RepReader=None,
                             rep_token: Union[str, int]=-1,
                             hidden_layers: Union[List[int], int]=-1,
                             component_index: int=0,
                             which_hidden_states: Optional[str]=None,
                             **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params =  {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]


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

        if self.image_processor:
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):
        return outputs

    def _forward(self, model_inputs, rep_token, hidden_layers, rep_reader=None, component_index=0, which_hidden_states=None, pad_token_id=None, device = "cuda"):
        """
        Args:
        - which_hidden_states (str): Specifies which part of the model (encoder, decoder, or both) to compute the hidden states from. 
                        It's applicable only for encoder-decoder models. Valid values: 'encoder', 'decoder'.
        """
        # get model hidden states and optionally transform them with a RepReader
        with torch.no_grad():
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input
            outputs =  self.model(**model_inputs, output_hidden_states=True)

        reasoning_hidden_states_layers, response_hidden_states_layers = self._get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)
        # ic(len(reasoning_hidden_states_layers)) # 27, both are 27
        if rep_reader is None:
            # Concatenate reasoning and response hidden states into a single tensor
            combined_hidden_states = torch.cat(
                [torch.stack(list(reasoning_hidden_states_layers.values())),
                torch.stack(list(response_hidden_states_layers.values()))],
                dim=0
            )
            # ic(combined_hidden_states[-1].shape) # 54 (num_layers * 2))
            return combined_hidden_states
        # if rep_reader is None:
        #     return {
        #     "hidden_states": hidden_states,
        #     "reasoning_hidden_states": reasoning_hidden_states_layers,
        #     "response_hidden_states": response_hidden_states_layers,
        # }
        # transformed_hidden_states = rep_reader.transform(hidden_states, hidden_layers, component_index, device) # transformed_hidden_states
        transformed_reasoning_hidden_states = rep_reader.transform(reasoning_hidden_states_layers, hidden_layers, component_index, device)
        transformed_response_hidden_states = rep_reader.transform(response_hidden_states_layers, hidden_layers, component_index, device)
        # ic(transformed_reasoning_hidden_states, transformed_response_hidden_states)
        # return {
        #     "transformed_hidden_states": transformed_hidden_states,
        #     "transformed_reasoning_hidden_states": transformed_reasoning_hidden_states,
        #     "transformed_response_hidden_states": transformed_response_hidden_states,
        #     }
        combined_transformed_hidden_states = torch.cat(
            [torch.stack(list(transformed_reasoning_hidden_states.values())),
            torch.stack(list(transformed_response_hidden_states.values()))],
            dim=0
        )
        ic(len(combined_transformed_hidden_states)) # 54 (num_layers * 2))
        return combined_transformed_hidden_states

    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args):
        # Wrapper method to get a dictionary of hidden states from a list of strings.

        """
        Wrapper method to get a dictionary of hidden states from a list of strings.
        Parameters:
        - train_inputs (str, List[str], List[List[str]]): List of strings to get hidden states for.
        - rep_token (str, int): Token to use as the representation.
        - hidden_layers (str, int): Which layer(s) to get hidden states for.
        - batch_size (int): Batch size to use.
        - which_hidden_states (str): Which part of the model to get hidden states from ('encoder', 'decoder', or None).
        - **tokenizer_args: Additional arguments to pass to tokenizer.
        Returns:
        - A dictionary where the keys are the layer(s) specified in hidden_layers, and the values are 2D numpy arrays of shape (n_samples, hidden_size).
        """
        
        # Call the pipeline with the provided inputs and parameters to get hidden states for the training inputs.
        hidden_states_outputs = self(
            train_inputs, 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            batch_size=batch_size, 
            rep_reader=None, 
            which_hidden_states=which_hidden_states, 
            **tokenizer_args
        )

        ic(len(hidden_states_outputs)) # 1024
        ic(hidden_states_outputs[0].shape) # [1, 32, 1024]
        ic(hidden_layers) # [-1, -2,...-27]
        # hidden_states_outputs = list_of_all_hidden_states["hidden_states"]
        # reasoning_hidden_states_outputs = list_of_all_hidden_states["reasoning_hidden_states"]
        # response_hidden_states_outputs = list_of_all_hidden_states["response_hidden_states"]
        # ic(hidden_states_outputs, reasoning_hidden_states_outputs, response_hidden_states_outputs)
        # Initialize a dictionary to store hidden states for each layer.
        hidden_states = {layer: [] for layer in hidden_layers}
        
        # Iterate over the batches of hidden states returned by the pipeline.
        for hidden_states_batch in hidden_states_outputs:
            # For each layer in the batch, append the hidden states to the corresponding list in the dictionary.
            for layer in hidden_states_batch:

                hidden_states[layer].extend(hidden_states_batch[layer])
        
        # Convert the lists of hidden states for each layer into 2D numpy arrays and return the dictionary.
        return {k: np.vstack(v) for k, v in hidden_states.items()}
    
    def _validate_params(self, n_difference, direction_method):
        # validate params for get_directions
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
            **tokenizer_args,):
        """Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """

        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]
        
        self._validate_params(n_difference, direction_method)

        # initialize a DirectionFinder
        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)

		# if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            hidden_states = self._batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args)
            ic("get_directions hidden states", hidden_states.shape)
            # hidden_states= list_of_all_states["hidden_states"]
            # reasoning_hidden_states_outputs = list_of_all_states["reasoning_hidden_states"]
            # response_hidden_states_outputs = list_of_all_states["response_hidden_states"]# get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]

		# get the directions
        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels)
        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
            hidden_states, train_labels, hidden_layers)
        
        return direction_finder
