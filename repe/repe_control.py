from transformers.pipelines import TextGenerationPipeline

# wrapping classes
import torch
import numpy as np


class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block  # Store the original block to be wrapped
        self.output = None  # Placeholder for storing the output of the block
        self.controller = None  # Placeholder for storing the controller (activations)
        self.mask = None  # Placeholder for storing the mask
        self.token_pos = None  # Placeholder for storing the token position
        self.normalize = False  # Flag to indicate whether normalization is applied

    def forward(self, *args, **kwargs):
        # Perform a forward pass through the original block
        output = self.block(*args, **kwargs)

        # Check if the output is a tuple (e.g., for models with multiple outputs)
        if isinstance(output, tuple):
            self.output = output[0]  # Store the first element of the tuple as the output
            modified = output[0]  # Use the first element for modification
        else:
            self.output = output  # Store the output directly
            modified = output  # Use the output for modification

        # If a controller is set, modify the output using the controller
        if self.controller is not None:
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)  # Compute the norm of the output before modification

            # If a mask is provided, use it
            if self.mask is not None:
                mask = self.mask
            # If no mask is provided, create one based on position IDs (if available)
            elif "position_ids" in kwargs:
                pos = kwargs["position_ids"]  # Extract position IDs from kwargs
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)  # Find the indices of padding tokens
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)  # Create column indices
                target_shape = modified.shape  # Get the shape of the modified output
                mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)  # Create the mask
                mask = mask.to(modified.dtype)  # Convert the mask to the same dtype as the output
            else:
                # If no position IDs are available, use a default mask of 1.0
                mask = 1.0

            # Ensure the controller has the correct shape
            if len(self.controller.shape) == 1:
                self.controller = self.controller.reshape(1, 1, -1)
            assert len(self.controller.shape) == len(modified.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."

            # Move the controller and mask to the same device as the output
            self.controller = self.controller.to(modified.device)
            if type(mask) == torch.Tensor:
                mask = mask.to(modified.device)

            # Apply the controller to the specified token positions
            if isinstance(self.token_pos, int):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple) or isinstance(self.token_pos, np.ndarray):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, str):
                if self.token_pos == "end":
                    len_token = self.controller.shape[1]  # Get the length of the controller
                    modified[:, -len_token:] = self.operator(modified[:, -len_token:], self.controller * mask)
                elif self.token_pos == "start":
                    len_token = self.controller.shape[1]  # Get the length of the controller
                    modified[:, :len_token] = self.operator(modified[:, :len_token], self.controller * mask)
                else:
                    assert False, f"Unknown token position {self.token_pos}."  # Raise an error for invalid token positions
            else:
                modified = self.operator(modified, self.controller * mask)  # Apply the operator to the entire output

            # If normalization is enabled, normalize the modified output
            if self.normalize:
                norm_post = torch.norm(modified, dim=-1, keepdim=True)  # Compute the norm after modification
                modified = modified / norm_post * norm_pre  # Scale the modified output to match the original norm

        # If the output is a tuple, replace the first element with the modified output
        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified  # Otherwise, replace the output directly

        return output  # Return the modified output

    def set_controller(self, activations, token_pos=None, masks=None, normalize=False, operator='linear_comb'):
        # Set the controller (activations) and related parameters
        self.normalize = normalize  # Set the normalization flag
        self.controller = activations.squeeze()  # Store the activations, removing unnecessary dimensions
        self.mask = masks  # Store the mask
        self.token_pos = token_pos  # Store the token position

        # Define the operator based on the specified type
        if operator == 'linear_comb':
            def op(current, controller):
                return current + controller  # Add the controller to the current output
        elif operator == 'piecewise_linear':
            def op(current, controller):
                sign = torch.sign((current * controller).sum(-1, keepdim=True))  # Compute the sign of the dot product
                return current + controller * sign  # Add the controller scaled by the sign
        elif operator == 'projection':
            def op(current, controller):
                raise NotImplementedError  # Projection operator is not implemented
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")  # Raise an error for unsupported operators
        self.operator = op  # Store the operator

    def reset(self):
        # Reset the block's state
        self.output = None  # Clear the stored output
        self.controller = None  # Clear the controller
        self.mask = None  # Clear the mask
        self.token_pos = None  # Clear the token position
        self.operator = None  # Clear the operator

    def set_masks(self, masks):
        # Set the mask for the block
        self.mask = masks


BLOCK_NAMES = [
    "self_attn",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm"
    ]
    
class WrappedReadingVecModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        # Initialize the wrapped model with the given model and tokenizer
        super().__init__()
        self.model = model  # Store the model
        self.tokenizer = tokenizer  # Store the tokenizer
        
    def forward(self, *args, **kwargs):
        # Forward pass through the underlying model
        return self.model(*args, **kwargs)
        
    def generate(self, **kwargs):
        # Generate text using the underlying model's generate method
        return self.model.generate(**kwargs)
        
    def get_logits(self, tokens):
        # Get the logits (raw predictions) for the given tokens
        with torch.no_grad():  # Disable gradient computation for inference
            logits = self.model(tokens.to(self.model.device)).logits  # Forward pass and extract logits
            return logits
        
    def run_prompt(self, prompt, **kwargs):
        # Run the model on a given prompt and return the output
        with torch.no_grad():  # Disable gradient computation for inference
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)  # Tokenize the prompt
            input_ids = inputs.input_ids.to(self.model.device)  # Move input IDs to the model's device
            attention_mask = inputs.attention_mask.to(self.model.device)  # Move attention mask to the model's device
            output = self.model(input_ids, attention_mask=attention_mask)  # Forward pass through the model
            return output
    
    def wrap(self, layer_id, block_name):
        # Wrap a specific block in a specific layer with WrappedBlock
        assert block_name in BLOCK_NAMES  # Ensure the block name is valid
        if self.is_wrapped(self.model.model.layers[layer_id]):  # Check if the layer is already wrapped
            block = getattr(self.model.model.layers[layer_id].block, block_name)  # Get the block
            if not self.is_wrapped(block):  # If the block is not wrapped, wrap it
                setattr(self.model.model.layers[layer_id].block, block_name, WrappedBlock(block))
        else:
            block = getattr(self.model.model.layers[layer_id], block_name)  # Get the block
            if not self.is_wrapped(block):  # If the block is not wrapped, wrap it
                setattr(self.model.model.layers[layer_id], block_name, WrappedBlock(block))

    def wrap_decoder_block(self, layer_id):
        # Wrap the entire decoder block in a specific layer with WrappedBlock
        block = self.model.model.layers[layer_id]  # Get the layer
        if not self.is_wrapped(block):  # If the layer is not wrapped, wrap it
            self.model.model.layers[layer_id] = WrappedBlock(block)

    def wrap_all(self):
        # Wrap all layers and blocks in the model
        for layer_id, layer in enumerate(self.model.model.layers):  # Iterate over all layers
            for block_name in BLOCK_NAMES:  # Iterate over all block names
                self.wrap(layer_id, block_name)  # Wrap each block
            self.wrap_decoder_block(layer_id)  # Wrap the decoder block

    def wrap_block(self, layer_ids, block_name):
        # Wrap specific blocks in specific layers
        def _wrap_block(layer_id, block_name):
            if block_name in BLOCK_NAMES:  # If the block name is valid, wrap it
                self.wrap(layer_id, block_name)
            elif block_name == 'decoder_block':  # If the block name is 'decoder_block', wrap the decoder block
                self.wrap_decoder_block(layer_id)
            else:
                assert False, f"No block named {block_name}."  # Raise an error for invalid block names

        if isinstance(layer_ids, (list, tuple, np.ndarray)):  # If layer_ids is a list, tuple, or array
            for layer_id in layer_ids:  # Iterate over all layer IDs
                _wrap_block(layer_id, block_name)  # Wrap each block
        else:
            _wrap_block(layer_ids, block_name)  # Wrap the single block

    def get_activations(self, layer_ids, block_name='decoder_block'):
        # Get activations from specific layers and blocks
        def _get_activations(layer_id, block_name):
            current_layer = self.model.model.layers[layer_id]  # Get the current layer

            if self.is_wrapped(current_layer):  # If the layer is wrapped
                current_block = current_layer.block  # Get the block
                if block_name == 'decoder_block':  # If the block name is 'decoder_block', return the layer's output
                    return current_layer.output
                elif block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_block, block_name)):  # If the block is wrapped
                    return getattr(current_block, block_name).output  # Return the block's output
                else:
                    assert False, f"No wrapped block named {block_name}."  # Raise an error for invalid block names

            else:  # If the layer is not wrapped
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_layer, block_name)):  # If the block is wrapped
                    return getattr(current_layer, block_name).output  # Return the block's output
                else:
                    assert False, f"No wrapped block named {block_name}."  # Raise an error for invalid block names
                
        if isinstance(layer_ids, (list, tuple, np.ndarray)):  # If layer_ids is a list, tuple, or array
            activations = {}  # Initialize a dictionary to store activations
            for layer_id in layer_ids:  # Iterate over all layer IDs
                activations[layer_id] = _get_activations(layer_id, block_name)  # Get activations for each layer
            return activations  # Return the dictionary of activations
        else:
            return _get_activations(layer_ids, block_name)  # Get activations for the single layer

    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=None, masks=None, normalize=False, operator='linear_comb'):
        # Set the controller for specific layers and blocks
        def _set_controller(layer_id, activations, block_name, masks, normalize, operator):
            current_layer = self.model.model.layers[layer_id]  # Get the current layer

            if block_name == 'decoder_block':  # If the block name is 'decoder_block'
                current_layer.set_controller(activations, token_pos, masks, normalize, operator)  # Set the controller
            elif self.is_wrapped(current_layer):  # If the layer is wrapped
                current_block = current_layer.block  # Get the block
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_block, block_name)):  # If the block is wrapped
                    getattr(current_block, block_name).set_controller(activations, token_pos, masks, normalize, operator)  # Set the controller
                else:
                    return f"No wrapped block named {block_name}."  # Return an error message for invalid block names

            else:  # If the layer is not wrapped
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_layer, block_name)):  # If the block is wrapped
                    getattr(current_layer, block_name).set_controller(activations, token_pos, masks, normalize, operator)  # Set the controller
                else:
                    return f"No wrapped block named {block_name}."  # Return an error message for invalid block names
                
        if isinstance(layer_ids, (list, tuple, np.ndarray)):  # If layer_ids is a list, tuple, or array
            assert isinstance(activations, dict), "activations should be a dictionary"  # Ensure activations is a dictionary
            for layer_id in layer_ids:  # Iterate over all layer IDs
                _set_controller(layer_id, activations[layer_id], block_name, masks, normalize, operator)  # Set the controller for each layer
        else:
            _set_controller(layer_ids, activations, block_name, masks, normalize, operator)  # Set the controller for the single layer
      
    def reset(self):
        # Reset all layers and blocks in the model
        for layer in self.model.model.layers:  # Iterate over all layers
            if self.is_wrapped(layer):  # If the layer is wrapped
                layer.reset()  # Reset the layer
                for block_name in BLOCK_NAMES:  # Iterate over all block names
                    if self.is_wrapped(getattr(layer.block, block_name)):  # If the block is wrapped
                        getattr(layer.block, block_name).reset()  # Reset the block
            else:  # If the layer is not wrapped
                for block_name in BLOCK_NAMES:  # Iterate over all block names
                    if self.is_wrapped(getattr(layer, block_name)):  # If the block is wrapped
                        getattr(layer, block_name).reset()  # Reset the block

    def set_masks(self, masks):
        # Set masks for all layers and blocks in the model
        for layer in self.model.model.layers:  # Iterate over all layers
            if self.is_wrapped(layer):  # If the layer is wrapped
                layer.set_masks(masks)  # Set masks for the layer
                for block_name in BLOCK_NAMES:  # Iterate over all block names
                    if self.is_wrapped(getattr(layer.block, block_name)):  # If the block is wrapped
                        getattr(layer.block, block_name).set_masks(masks)  # Set masks for the block
            else:  # If the layer is not wrapped
                for block_name in BLOCK_NAMES:  # Iterate over all block names
                    if self.is_wrapped(getattr(layer, block_name)):  # If the block is wrapped
                        getattr(layer, block_name).set_masks(masks)  # Set masks for the block

    def is_wrapped(self, block):
        # Check if a block is wrapped
        if hasattr(block, 'block'):  # If the block has a 'block' attribute, it is wrapped
            return True
        return False  # Otherwise, it is not wrapped
    
    def unwrap(self):
        # Unwrap all layers and blocks in the model
        for l, layer in enumerate(self.model.model.layers):  # Iterate over all layers
            if self.is_wrapped(layer):  # If the layer is wrapped
                self.model.model.layers[l] = layer.block  # Unwrap the layer
            for block_name in BLOCK_NAMES:  # Iterate over all block names
                if self.is_wrapped(getattr(self.model.model.layers[l], block_name)):  # If the block is wrapped
                    setattr(self.model.model.layers[l],
                            block_name,
                            getattr(self.model.model.layers[l], block_name).block)  # Unwrap the block

class RepControlPipeline(TextGenerationPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers, 
                 block_name="decoder_block", 
                 control_method="reading_vec",
                 **kwargs):
        
        # Ensure the control method is "reading_vec", as no other methods are supported yet
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        
        # Ensure the block_name is "decoder_block" or the model architecture supports it
        assert block_name == "decoder_block" or "LlamaForCausalLM" in model.config.architectures, f"{model.config.architectures} {block_name} not supported yet"
        
        # Wrap the model and tokenizer in a custom WrappedReadingVecModel
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        
        # Unwrap any previously wrapped layers in the model
        self.wrapped_model.unwrap()
        
        # Wrap specific layers and blocks in the model for control
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        
        # Store the block name and layers for later use
        self.block_name = block_name
        self.layers = layers

        # Initialize the parent TextGenerationPipeline with the model and tokenizer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def __call__(self, text_inputs, activations=None, **kwargs):
        # If activations are provided, reset the model and set the controller with the activations
        if activations is not None:
            self.wrapped_model.reset()  # Reset the wrapped model to clear any previous state
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)  # Set the controller for the specified layers and block

        # Call the parent pipeline's __call__ method to generate outputs
        outputs = super().__call__(text_inputs, **kwargs)
        
        # Reset the wrapped model after generating outputs to clear the state
        self.wrapped_model.reset()

        # Return the generated outputs
        return outputs
