import keras.backend as K
from keras.layers.core import RepeatVector, TimeDistributedDense, MaskedLayer
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
        if self.first_step:
            state = K.dot(x, self.W) + self.b
            self.first_step = False
        else:
            state = K.dot(prev_output, self.U)
        output = self.activation(state)
        #output = self.activation(h + K.dot(prev_output, self.U))
        return output, [output]

def get_rnn_model(dshape, num_output_vars, macro_dims=10, discount=0.8, output_type='bool', archtype='RNN', 
                  hidden_layer_dims=[], activation='tanh', optimizer='rmsprop'):

    num_samples, num_timesteps, num_input_vars = dshape
    print "Creating %s-valued model, archtype %s, internal activation=%s, optimizer=%s" % \
          (output_type, archtype, activation, optimizer)
    print "num_samples=%d, num_timesteps=%d, num_input_vars=%d, num_output_vars=%d" % (num_samples, num_timesteps, num_input_vars, num_output_vars)
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
    
    c_activation, extra_activation_cls = activation, None
    if activation.lower() in ['prelu', 'leakyrelu']:
        c_activation = 'linear'
        if activation.lower() == 'prelu':
            from keras.layers.advanced_activations import PReLU
            extra_activation_cls = PReLU
        else:
            from keras.layers.advanced_activations import LeakyReLU
            extra_activation_cls = LeakyReLU
        
    init_dist = 'lecun_uniform'
    init_dist = 'he_normal'
    
    if archtype == 'RNN':
        if hidden_layer_dims:
            raise Exception('hidden_layer_dims not supported')
            
        model.add(SimpleRNN(macro_dims, input_dim=num_input_vars, return_sequences=True, input_length=num_timesteps, 
                            activation=c_activation, init=init_dist))
        if extra_activation_cls is not None:
            model.add(extra_activation_cls())
        model.add(TimeDistributedDense(num_output_vars, activation=output_activation))
        
    elif archtype == 'RNNInitialTime':
        c_dim = num_input_vars
        for d in hidden_layer_dims:
            model.add(TimeDistributedDense(d, input_dim=c_dim, input_length=num_timesteps, 
                                           activation=c_activation, init=init_dist))
            c_dim = d
            if extra_activation_cls is not None:
                model.add(extra_activation_cls())
            
        model.add(SimpleRNNSingleInput(macro_dims, input_dim=c_dim, return_sequences=True, 
                                       input_length=num_timesteps, 
                                       activation=c_activation, init=init_dist))
        if extra_activation_cls is not None:
            model.add(extra_activation_cls())
            
        c_dim = macro_dims
        for d in hidden_layer_dims[::-1]:
            model.add(TimeDistributedDense(d, input_dim=c_dim, input_length=num_timesteps, 
                                           activation=c_activation, init=init_dist))
            c_input_dim = d
            if extra_activation_cls is not None:
                model.add(extra_activation_cls())

        model.add(TimeDistributedDense(num_output_vars, input_dim=c_dim, 
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
    model.compile(class_mode='binary', loss='mse', optimizer=optimizer)

    return model
