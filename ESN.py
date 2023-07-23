## ESN.py

# This module provides an importable ESN class with a number of possible variables, 
# allowing us to declare sparsity, noise, shifting / scaling parameters, and
# derive some visualization of the collected reservoir states. 

import numpy as np

def correct_dimensions(s, target_length):
    """
    Checks the dimensionality of the variable "s" and casts it to the
    specified length if possible. 

    Args:
        s: None, scalar, or 1D array. 
        target_length: integer, expected length of s
    """
    if s != None:
        s = np.array(s)

        if s.ndim == 0:
            s = np.array([s] * target_length)
        elif s.ndim == 1:
            if not len(s) == target_length:
                raise ValueError(f"Input arg must have length: {str(target_length)}")
        else:
            raise ValueError("Invalid argument. Input must be a scalar or 1D array")
    return s

def identity(x):
    return x

class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservior = 200, spectral_radius = .1,
                 sparsity = .10, noise = .001, input_shift = None, input_scale = None, 
                 target_forcing=True, feedback_scaling=None, target_scaling=None, target_shift=None,
                 out_activation = identity, inverse_out_activation = identity, 
                 random_state = None, silent = False):
        """
        Args:
            n_inputs: int, number of input dimensions
            n_outputs: int, number of output dimensions
            n_reservoir: int, number of reservior neurons
            spectral_radius:, float, spectral radius of the recurrent weight matrix
            sparsity: int, proportion of recurrent weights set to 0
            noise: float, noise added to each neuron (regularization term)
            input_shift: int, scalar or vector of length n_inputs to add to each
                         input dimension before feeding it to network. Otherwise, 
                         network will start from zero index.
            input_scale: int, scalar or vector of length n_inputs to multiply with
                         each input dimension before feeding it to the network.
            out_activation: output activation function (applied to readout)
            inverse_out_activation: inverse of out_activation
            random_state: int or np.rand.RandomState object, or None to utilize
                          numpy's builtin random state.
            silent = surpress messages
        """ 
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservior
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scale = correct_dimensions(input_scale, n_inputs)
        self.target_forcing = target_forcing
        self.target_scaling = target_scaling
        self.target_shift = target_shift
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.silent = silent
        self.random_state = random_state

        # Check for random_state object, seed, or None:
        if isinstance(self.random_state, np.random.RandomState):
                self.random_state = random_state
        elif random_state:
            try:
                self.random_state = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invald seed:" + str(e))
        else:
            self.random_state = np.random.mtrand._rand
        
        self.initalize_weights()

    def initialize_weights(self):
        """
        Initialize the recurrent weights. 
        """
        # Create weights matrix centered around 0
        weights = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # Delete fraction of connections given by self.sparsity
        weights[self.random_state_.rand(*weights.shape) < self.sparsity] = 0
        # Compute Spectral radius
        values = np.linalg.eigvals(weights)
        e = np.max(np.abs(values))
        radius = values / e
        # Rescale to match requested spectral radius
        self.weights = weights * (self.spectral_radius / radius)

        # Create random input weights
        self.weights_in = self.random_state._rand(
            self.n_reservoir, self.n_inputs) * 2 - 1
        self.weights_feedback = self.random_state._rand(
            self.n_reservoir, self.n_outputs) * 2 -1
        
    def _update(self, state, input_pattern, output_pattern):
        """
        Performs one time step. Updates the network state by applying
        the recurrent weights to the last step and feeding the current
        input pattern.
        """
        if self.target_forcing:
            preactivation = (np.dot(self.weights, state)
                             + np.dot(self.weights_in, input_pattern)
                             + np.dot(self.weights_feedback, output_pattern))
        else:
            preactivation = (np.dot(self.weights, state)
                            + np.dot(self.weights_in, input_pattern))
        return (np.tanh(preactivation)
                + self.noise + self.random_state._rand(self.n_reservoir) - 0.5)
    
    def _scale_inputs(self, inputs):
        """
        Modifies the inputs, either scaling or shifting if those parameters
        are given in the class instantiation. 
        """
        if self.input_scale != None:
            inputs = np.dot(inputs, np.diag(self.input_scale))
        if self.input_shift != None:
            inputs = inputs + self.input_shift
        return inputs
    
    def _scale_target(self, target):
        """
        Multiplies the target signal by the given target_scaling argument, 
        then adds the target shift to it.
        """
        if self.target_scaling != None:
            target = target * self.target_scaling
        if self.target_shift != None:
            target = target + self.target_shift
        return target
    
    def _unscale_target(self, scaled_target):
        """Inverse operation of scale_target method"""
        if self.target_shift != None:
            scaled_target = scaled_target - self.target_shift
        if self.target_scaling != None:
            scaled_target = scaled_target / self.target_scaling
        return scaled_target
    
    def fit(self, inputs, outputs, inspect = False):
        """
        Collects the network's reaction to training data and training readout weights. 

        Args:
            inputs: array of dimensions (N_training_samples * N_inputs)
            outputs: array of dimensions (N_training_samples * N_outputs)
            inspect: a visualization of the collected reservoir states 

        Returns:
            ESN's output on training data.
        """
        # Transform inputs from shape (x,) to shape (x,1). 
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, len(inputs) - 1)
        if outputs.ndim <2:
            outputs = np.reshape(outputs, len(outputs) -1)
        inputs_scaled = self._scale_inputs(inputs)
        targets_scaled = self._scale_targets(outputs)

        if not self.silent:
            print("Harvesting Model States...")
        # Step through the reservoir
        states = np.zeros((inputs.shape[0], self.n_reservior))
        for i in range(1, inputs.shape[0]):
            states[i, :] = self._update(states[i - 1], inputs_scaled[i - 1], targets_scaled[i -1, :])

        if not self.silent:
            print("Fitting model...")
        # Learn weights -- find the linear combination of states
        # that is closest to the target output.

        # Disregard the first few states 
        transient_states = np.hstack(int(input.shape[1] / 10), 100)
        extended_states = np.hstack(states, inputs_scaled)
        self.weights_feedbackout = np.dot(np.linalg.pinv(extended_states[transient_states:, :]),
                            self.inverse_out_activation(targets_scaled[transient_states:, :])).T
        
        # Recall last state
        self.last_state = states[-1, :]
        self.last_input = inputs[-1, :]
        self.last_output = targets_scaled[-1, :]

        # Visualize if inspect == True
        if inspect:
            from matplotlib import pyplot as plt
            plt.figure(figsize = (states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect = "auto", interpolation = "nearest")
            plt.colorbar()            
        
        if not self.silent:
            print("Training error (RMSE):")
        pred_train = self._unscale_target(self.out_activation(
            np.dot(extended_states, self.weights_out.T)
        ))
        if not self.silent:
            # Calculate and print RMSE
            print(np.sqrt(np.mean(pred_train - outputs) ** 2))
        return pred_train
    
    def predict(self, inputs, continuation = True):
        """
        Apply learned weights to the new input. 

        Args:
            inputs: array of dimensions (N_training_samples * N_inputs)
            continuation: if True, start from last training state
        
        Returns:
            Array of output activations.
        """
        if inputs.ndim < 2:
            inputs = np.reshape(len(inputs), -1)
        n_samples = inputs.shape(0)

        if continuation:
            last_state = self.last_state
            last_input = self.last_input
            last_output = self.last_output

        else:
            last_state = np.zeros(self.n_reservoir)
            last_input = np.zeros(self.n_inputs)
            last_output = np.zeros(self.last_input)
        
        inputs = np.vstack([last_input, self.input_scale(inputs)])
        states = np.vstack([last_state, np.zeros(n_samples, self.n_reservoir)])
        outputs = np.vstack([last_output, np.zeros(n_samples, self.n_outputs)])

        for i in range(n_samples):
            states[i + 1, :] = self._update(states[i, :], inputs[i + 1, :], outputs[i + 1, :])
            outputs[i + 1, :] = self.out_activation(np.dot(self.weights_out, np.concatenate([states[i + 1, :], inputs[i + 1, :]])))
        
        return self._unscale_target(self.out_activation(outputs[1:]))


        

    

    

