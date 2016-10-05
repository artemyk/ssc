import keras.backend as K
from keras.layers.core import RepeatVector, TimeDistributedDense
import numpy as np
            
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, RepeatVector
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import SGD

from keras.objectives import binary_crossentropy, mean_squared_error

class PadZerosVector(RepeatVector):
    '''Pad the input with n-1 zeros
    # Input shape
        2D tensor of shape `(nb_samples, features)`.
    # Output shape
        3D tensor of shape `(nb_samples, n, features)`.
    # Arguments
        n: integer, repetition factor.
    '''
    def get_output(self, train=False):
        x = self.get_input(train)
        x_shape = K.shape(x)
        #stacked = K.concatenate( [K.reshape(x, (x_shape[0], x_shape[1], 1)), K.zeros( (x_shape[0], x_shape[1], self.n-1) )], axis=2 )
        K.zeros( (x_shape[0], x_shape[1], self.n-1) )
        #stacked = K.concatenate( [K.reshape(x, (x_shape[0], x_shape[1], 1)), K.zeros( (x_shape[0], x_shape[1], self.n-1) )], axis=2 )
        return stacked.dimshuffle((0, 2, 1))

class SimpleRNNSingleInput(SimpleRNN):
    def get_output(self, *kargs, **kwargs):
        self.first_step = True
        return super(SimpleRNNSingleInput, self).get_output(*kargs, **kwargs)

    def step(self, x, states):
        # states only contains the previous output.
        assert len(states) == 1
        prev_output = states[0]
        mult = 1. if self.first_step else 0.
        
        state = mult * (K.dot(x, self.W) + self.b) + (1.-mult) * K.dot(prev_output, self.U)
        self.first_step = False
        
        #if self.first_step:
        #    state = K.dot(x, self.W) + self.b
        #    self.first_step = False
        #else:
        #    state = K.dot(prev_output, self.U)
        output = self.activation(state)
        #output = self.activation(h + K.dot(prev_output, self.U))
        return output, [output]

class SimpleRNNSingleInput2(SimpleRNN):
    def step(self, x, states):
        assert len(states) == 1
        prev_output = states[0]
        state = 0.0*(K.dot(x, self.W)) + K.dot(prev_output, self.U) + self.b
        output = self.activation(state)
        return output, [output]

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        #initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        #initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        #reducer = K.zeros((self.input_dim, self.output_dim))
        #initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        #c_mask = K.zeros_like(X)  # (samples, timesteps, input_dim)
        #c_mask[:,0,:] = 0
        #return X # K.dot(X, self.W) + self.b
        initial_state = X[:,0,:]  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states
        
        

class SimpleRNNTDD(SimpleRNN):
    def __init__(self, *kargs, **kwargs):
        super(SimpleRNNTDD, self).__init__(*kargs, **kwargs)
        self.I = K.variable(value=np.eye(self.input_dim))
        
    def step(self, x, states):
        prev_output = states[0]
        h = K.dot(x, 0.0*self.W + 1.0*self.I) + self.b
        output = self.activation(h + K.dot(prev_output, self.U))
        return output, [output]
    
class TDD(TimeDistributedDense):
    def __init__(self, *kargs, **kwargs):
        super(TDD, self).__init__(*kargs, **kwargs)
        mult = np.zeros(self.input_length)
        mult[0] = 1.0
        t = K.variable(value=mult)
        t = K.expand_dims(t, 0)
        t = K.expand_dims(t, 2)
        self.mult = t
        
    def get_output(self, train=False):
        y = super(TDD, self).get_output(train)
        return self.mult * y
    
        """
        X = self.get_input(train)

        output1 = K.dot(X[0], self.W) + self.b
        x_shape = K.shape(X)
        z=K.zeros( (x_shape[0], x_shape[1], self.input_length-1) )
        outputs = K.concatenate([output1[:,None,:],] + [z,], axis=1)
        #cmask[:,0,:]+=1.0
        outputs = self.activation(outputs)
        return outputs
        """

    
def get_rnn_model(num_timesteps, num_input_vars, num_output_vars, macro_dims=10, discount=0.8, 
                  output_type='bool', 
                  archtype='RNN', 
                  hidden_layer_dims=[],
                  hidden_activation='tanh', 
                  macro_activation=None,
                  activation_props=dict(),
                  optimizer='rmsprop',
                 regularization=0.0):

    if macro_activation is None:
        macro_activation = hidden_activation
    print "Creating %s-valued model, archtype %s, hidden activation=%s, macro activation=%s, optimizer=%s" % \
          (output_type, archtype, hidden_activation, macro_activation, optimizer)
    print "num_timesteps=%d, num_input_vars=%d, num_output_vars=%d" % (num_timesteps, num_input_vars, num_output_vars)
    print "macro_dims=%d, hidden_layer_dims=%s" % (macro_dims, str(hidden_layer_dims))
    print "discount factor=%0.2f" % discount
    
    """
    class Copy(MaskedLayer):
        def get_output(self, train=False):
            X = self.get_input(train)
            return X

    class PadZerosVector2(RepeatVector):
        def get_output(self, train=False):
            x = self.get_input(train)
            x[:,1:,:]=0.0
            return x
            #stacked = K.concatenate( [K.reshape(x, (x_shape[0], x_shape[1], 1)), K.zeros( (x_shape[0], x_shape[1], self.n-1) )], axis=2 )
            #print self.input_length, x_shape[0]
            #print self.input_
            #K.zeros( (x_shape[0], x_shape[1], self.input_length-1) )
            z = x * 0.0
            #x_shape = (num_vars, num_samples)
            #x2 = K.concatenate( [x,] + z * (self.input_length-1) , axis=3)
            stacked = K.concatenate( [K.expand_dims(x), K.expand_dims(x)], axis=2 )
            return K.permute_dimensions(stacked, (0, 2, 1))        

    """


    OUTPUT_TYPES = ('bool', 'real')
    if output_type not in OUTPUT_TYPES:
        print 'output_type [%s] must be one of %s' % (output_type, OUTPUT_TYPES)

    if output_type == 'bool':
        output_activation = 'sigmoid'
    else:
        output_activation = 'linear'

    model = Sequential()
    
    def get_activation(p_act):
        act = p_act.lower()
        if act == 'prelu':
            from keras.layers.advanced_activations import PReLU
            act = PReLU(**activation_props)
        elif act == 'srelu':
            from keras.layers.advanced_activations import SReLU
            act = SReLU(**activation_props)
        elif act == 'leakyrelu':
            from keras.layers.advanced_activations import LeakyReLU
            act = LeakyReLU(**activation_props)
        return act
        
    hidden_activation = get_activation(hidden_activation)
    macro_activation  = get_activation(macro_activation)
        
    init_dist = 'lecun_uniform'
    init_dist = 'he_normal'
    
    regobj = None
    if regularization:
        from keras.regularizers import l1
        regobj = l1(l=regularization)

    if archtype == 'rnn':
        if hidden_layer_dims: raise Exception('hidden_layer_dims not supported')
            
        model.add(SimpleRNN(macro_dims, input_dim=num_input_vars, return_sequences=True, input_length=num_timesteps, 
                            activation=macro_activation, init=init_dist, U_regularizer=regobj))
        
        model.add(TimeDistributedDense(num_output_vars, activation=output_activation, init=init_dist))
        """
    elif archtype == 'rnn_identity':
        if hidden_layer_dims:
            raise Exception('hidden layer dis should be set to None for %s' % archtype)
        if macro_dims != num_input_vars:
            print "Setting macro_dims=num_input_vars for rnn_identity"
            macro_dims = num_input_vars
            # raise Exception('macro_dims should equal num_input_vars for %s'% archtype)
            
        model.add(TimeDistributedDense(num_input_vars, input_dim=num_input_vars, input_length=num_timesteps, 
                                       activation='linear', init='identity', trainable=False))
            
        model.add(SimpleRNNSingleInput(num_input_vars, input_dim=num_input_vars, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        model.add(TimeDistributedDense(num_input_vars, input_dim=num_input_vars, input_length=num_timesteps, 
                                       activation='linear', init='identity', trainable=False))
            
    elif archtype == 'rnn_init_time':
        c_dim = num_input_vars
        for d in hidden_layer_dims:
            model.add(TimeDistributedDense(d, input_dim=c_dim, input_length=num_timesteps, 
                                           activation=hidden_activation, init=init_dist))
            c_dim = d
            
        model.add(SimpleRNNSingleInput(macro_dims, input_dim=c_dim, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        c_dim = macro_dims
        for d in hidden_layer_dims[::-1]:
            model.add(TimeDistributedDense(d, input_dim=c_dim, input_length=num_timesteps, 
                                           activation=hidden_activation, init=init_dist))
            c_input_dim = d

        model.add(TimeDistributedDense(num_output_vars, input_dim=c_dim, 
                                       activation=output_activation, init=init_dist))
    """
    elif archtype == 'rnn1':
        model.add(TimeDistributedDense(macro_dims, input_dim=num_input_vars, input_length=num_timesteps, 
                                       activation='linear', init=init_dist))
            
        model.add(SimpleRNNSingleInput2(macro_dims, U_regularizer=regobj, input_dim=macro_dims, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        model.add(TimeDistributedDense(num_output_vars, input_dim=macro_dims, 
                                       activation=output_activation, init=init_dist))      
    elif archtype == 'rnn2':
        model.add(TDD(macro_dims, input_dim=num_input_vars, input_length=num_timesteps, 
                                       activation='linear', init=init_dist))
            
        model.add(SimpleRNNTDD(macro_dims, U_regularizer=regobj, input_dim=macro_dims, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        model.add(TimeDistributedDense(num_output_vars, input_dim=macro_dims, 
                                       activation=output_activation, init=init_dist))  
        
    elif archtype == 'rnn2_stacked':
        model.add(TDD(100, input_dim=num_input_vars, input_length=num_timesteps, 
                                       activation='tanh', init=init_dist))
        model.add(TDD(50, input_dim=100, input_length=num_timesteps, 
                                       activation='tanh', init=init_dist))
        model.add(TDD(macro_dims, input_dim=50, input_length=num_timesteps, 
                                       activation='tanh', init=init_dist))
            
        model.add(SimpleRNNTDD(macro_dims, U_regularizer=regobj, input_dim=macro_dims, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        model.add(TimeDistributedDense(num_output_vars, input_dim=macro_dims, 
                                       activation=output_activation, init=init_dist))       
        
    elif archtype == 'rnn2_invstacked':
        model.add(TDD(macro_dims, input_dim=num_input_vars, input_length=num_timesteps, 
                                       activation='linear', init=init_dist))
        model.add(SimpleRNNTDD(macro_dims, U_regularizer=regobj, input_dim=macro_dims, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        model.add(TimeDistributedDense(50, input_dim=macro_dims, input_length=num_timesteps, 
                                       activation='tanh', init=init_dist))
        model.add(TimeDistributedDense(100, input_dim=50, input_length=num_timesteps, 
                                       activation='tanh', init=init_dist))
        model.add(TimeDistributedDense(num_output_vars, input_dim=100, 
                                       activation=output_activation, init=init_dist))          
        
    elif archtype == 'rnn3':
        model.add(TDD(macro_dims, input_dim=num_input_vars, input_length=num_timesteps, 
                                       activation='tanh', init=init_dist))
            
        model.add(SimpleRNNTDD(macro_dims, U_regularizer=regobj, input_dim=macro_dims, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        model.add(TimeDistributedDense(num_output_vars, input_dim=macro_dims, 
                                       activation=output_activation, init=init_dist))          
    elif archtype == 'rnn4':
        model.add(TDD(macro_dims, input_dim=num_input_vars, input_length=num_timesteps, 
                                       activation='tanh', init=init_dist))
            
        model.add(SimpleRNN(macro_dims, U_regularizer=regobj, input_dim=macro_dims, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        model.add(TimeDistributedDense(num_output_vars, input_dim=macro_dims, 
                                       activation=output_activation, init=init_dist))   
        
    elif archtype == 'rnn4_stacked':
        
        model.add(TimeDistributedDense(1000, input_dim=num_input_vars, input_length=num_timesteps, activation='tanh', init=init_dist))
        model.add(TDD(macro_dims, input_dim=1000, input_length=num_timesteps, 
                                       activation='tanh', init=init_dist))
            
        model.add(SimpleRNN(macro_dims, U_regularizer=regobj, input_dim=macro_dims, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=macro_activation, init=init_dist))
            
        model.add(TimeDistributedDense(num_output_vars, input_dim=macro_dims, 
                                       activation=output_activation, init=init_dist))    
        
    else:
        raise Exception('Unknown RNN architecture %s'% archtype)
    """    
    elif False:
        model.add(Dense(num_vars, input_dim=num_vars, activation='tanh')) # , init='uniform', activation='sigmoid'))
        model.add(RepeatVector(num_timesteps))
    elif False:
        #activation = 'relu'
        activation = 'tanh'
        #model.add(Copy(input_dim=num_vars))
        model.add(PadZerosVector2(num_timesteps, input_shape=(num_vars,num_timesteps)))
        model.add(SimpleRNN(hidden_dims, return_sequences=True, input_length=num_timesteps, activation=activation))
        model.add(TimeDistributedDense(num_vars, activation=output_activation))
        #model.add(Activation('sigmoid'))
    elif False:
        model.add(Dense(hidden_dims, input_dim=num_vars, activation='tanh')) # , init='uniform', activation='sigmoid'))
        model.add(PadZerosVector(num_timesteps))
        model.add(SimpleRNN(hidden_dims, return_sequences=True, input_length=num_timesteps, activation='tanh'))
        model.add(TimeDistributedDense(num_vars, activation=output_activation))
        #model.add(Activation('sigmoid'))
    elif False:
        #activation = 'relu'
        activation = 'tanh'
        model.add(TimeDistributedDense(hidden_dims, input_dim=num_vars, input_length=num_timesteps, activation=activation))
        model.add(SimpleRNN(hidden_dims, return_sequences=True, input_length=num_timesteps, activation=activation))
        model.add(TimeDistributedDense(num_vars, activation=output_activation))
        #model.add(Activation('sigmoid'))
    else:
    """
    
    if output_type == 'bool':
        loss_func = binary_crossentropy
    else:
        loss_func = lambda y_true, y_pred: (y_true - y_pred)**2.0 # mean_squared_error
        
    if True:
        timeweights = discount ** np.arange(num_timesteps)
        def get_timeweighted_loss(timeweights):
            def f(y_true, y_pred):
                ce = loss_func(y_pred, y_true)
                ce = timeweights * ce
                return K.mean(ce, axis=-1)
            return f
        
        loss = get_timeweighted_loss(timeweights[None,:,None])

    #loss = 'binary_crossentropy'
    print "Just doing MSE error, not timediscounted!"
    model.compile(loss='mse', optimizer=optimizer)

    return model
