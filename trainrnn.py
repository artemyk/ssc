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
            h = K.dot(x, self.W) + self.b
            self.first_step = False
        else:
            h = 0.0
        output = self.activation(h + K.dot(prev_output, self.U))
        return output, [output]

def get_rnn_model(dshape, hidden_dims=10, discount=0.8, output_type='bool'):

    num_samples, num_timesteps, num_vars = dshape
    print "Creating %s-valued model, num_samples=%d, num_timesteps=%d, num_vars=%d" % (output_type, num_samples, num_timesteps, num_vars)
    print "Discount factor=%0.2f" % discount
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
        loss_func = binary_crossentropy
    else:
        output_activation = 'linear'
        loss_func = lambda y_true, y_pred: (y_true - y_pred)**2 # mean_squared_error

    model = Sequential()
    if False:
        model.add(Dense(num_vars, input_dim=num_vars, activation='tanh')) # , init='uniform', activation='sigmoid'))
        model.add(RepeatVector(num_timesteps))
    elif True:
        model.add(TimeDistributedDense(hidden_dims, input_dim=num_vars, input_length=num_timesteps, activation=activation))
        model.add(SimpleRNNSingleInput(hidden_dims, input_dim=hidden_dims, return_sequences=True, input_length=num_timesteps, activation='tanh'))
        model.add(TimeDistributedDense(num_vars, activation=output_activation))
    elif True:
        model.add(SimpleRNNSingleInput(hidden_dims, input_dim=num_vars, return_sequences=True, input_length=num_timesteps, activation='tanh'))
        model.add(TimeDistributedDense(num_vars, activation=output_activation))
    elif True:
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
        #from keras.layers.advanced_activations import ELU, LeakyReLU
        #
        model.add(SimpleRNN(hidden_dims, input_dim=num_vars, return_sequences=True, input_length=num_timesteps, activation='tanh'))
        model.add(TimeDistributedDense(num_vars, activation=output_activation))


    #import theano
    #theano.config.optimizer='None' # 'fast_compile'
    #theano.config.exception_verbosity='high'

    #optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = 'sgd'
    optimizer = 'rmsprop'

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
