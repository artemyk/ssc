import keras.backend as K
from keras.layers.core import RepeatVector
import numpy as np

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
        #x_shape = K.shape(x)
        #stacked = K.concatenate( [K.reshape(x, (x_shape[0], x_shape[1], 1)), K.zeros( (x_shape[0], x_shape[1], self.n-1) )], axis=2 )
        K.zeros( (x_shape[0], x_shape[1], self.n-1) )
        #stacked = K.concatenate( [K.reshape(x, (x_shape[0], x_shape[1], 1)), K.zeros( (x_shape[0], x_shape[1], self.n-1) )], axis=2 )
        return stacked.dimshuffle((0, 2, 1))

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, RepeatVector
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import SGD

from keras.objectives import binary_crossentropy, mean_squared_error

def get_rnn_model(dshape, hidden_dims=10, discount=0.8, output_type='bool'):

    OUTPUT_TYPES = ('bool', 'real')
    if output_type not in OUTPUT_TYPES:
        print 'output_type [%s] must be one of %s' % (output_type, OUTPUT_TYPES)

    if output_type == 'bool':
        output_activation = 'sigmoid'
        loss_func = binary_crossentropy
    else:
        output_activation = 'linear'
        loss_func = binary_crossentropy # mean_squared_error

    num_samples, num_timesteps, num_vars = dshape
    print "Creating %s-valued model, num_samples=%d, num_timesteps=%d, num_vars=%d" % (output_type, num_samples, num_timesteps, num_vars)
    print "Discount factor=%0.2f" % discount

    model = Sequential()
    if False:
        model.add(Dense(num_vars, input_dim=num_vars, activation='tanh')) # , init='uniform', activation='sigmoid'))
        model.add(RepeatVector(num_timesteps))
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
    optimizer = 'sgd'
    optimizer = 'rmsprop'

    if True:
        timeweights = discount ** np.arange(num_timesteps)
        def get_timeweighted_loss(timeweights):
            def f(y_true, y_pred):
                ce = K.binary_crossentropy(y_pred, y_true)
                ce = timeweights * ce
                return K.mean(ce, axis=-1)
            return f
        
        loss = get_timeweighted_loss(timeweights[None,:,None])

    #loss = 'binary_crossentropy'
    model.compile(class_mode='binary', loss='mse', optimizer=optimizer)

    return model
