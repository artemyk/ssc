import time
import numpy as np
import keras.backend as K

import logging    
logging.getLogger('keras').setLevel(logging.INFO)

import dynpy

bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.budding_yeast_bn)

hidden_dims = 15
timesteps   = 10

start_time = time.time()

#from theano import tensor as T
from keras.layers.core import RepeatVector

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
        stacked = T.concatenate( [x.reshape((x.shape[0], x.shape[1], 1)), T.zeros( (x.shape[0], x.shape[1], self.n-1) )], axis=2 )
        return stacked.dimshuffle((0, 2, 1))

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, RepeatVector
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import SGD
# (nsamples, ntimesteps, nfeatures)

model = Sequential()
if False:
    model.add(Dense(bn.num_vars, input_dim=bn.num_vars, activation='tanh')) # , init='uniform', activation='sigmoid'))
    model.add(RepeatVector(timesteps))
elif False:
    model.add(Dense(hidden_dims, input_dim=bn.num_vars, activation='tanh')) # , init='uniform', activation='sigmoid'))
    model.add(PadZerosVector(timesteps))
    model.add(SimpleRNN(hidden_dims, return_sequences=True, input_length=timesteps, activation='tanh'))
    model.add(TimeDistributedDense(bn.num_vars, activation='sigmoid'))
    #model.add(Activation('sigmoid'))
elif True:
    model.add(TimeDistributedDense(hidden_dims, input_dim=bn.num_vars, input_length=timesteps, activation='relu'))
    model.add(SimpleRNN(hidden_dims, return_sequences=True, input_length=timesteps, activation='relu'))
    model.add(TimeDistributedDense(bn.num_vars, activation='sigmoid'))
    #model.add(Activation('sigmoid'))
else:
    from keras.layers.advanced_activations import ELU, LeakyReLU
    #
    model.add(SimpleRNN(hidden_dims, input_dim=bn.num_vars, return_sequences=True, input_length=timesteps, activation='tanh'))
    model.add(TimeDistributedDense(bn.num_vars, activation='sigmoid'))


#import theano
#theano.config.optimizer='None' # 'fast_compile'
#theano.config.exception_verbosity='high'

#optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = 'sgd'
optimizer = 'rmsprop'
loss      = 'binary_crossentropy'

if True:
    delta = 0.8
    #timeweights = np.zeros(timesteps, dtype='float32')
    timeweights = delta ** np.arange(timesteps)
    def get_timeweighted_binary_crossentropy(timeweights):
        def f(y_true, y_pred):
            ce = K.binary_crossentropy(y_pred, y_true)
            ce = timeweights * ce
            return K.mean(ce, axis=-1)
        return f
    
    loss = get_timeweighted_binary_crossentropy(timeweights[None,:,None])

#loss = 'binary_crossentropy'
model.compile(class_mode='binary', loss=loss, optimizer=optimizer)



N = 1000

trnN = int(.7*N)

trajs = np.concatenate([
         bn.get_trajectory(start_state=(np.random.rand(bn.num_vars) > 0.5).astype('uint8'), max_time=timesteps)[:,:,None]
         for _ in range(N)], 2).astype('float32').transpose([2,0,1])
#trajs -= 0.5
trajs_train = trajs[:trnN,:,:]
trajs_test  = trajs[trnN:,:,:]

X_train = trajs_train - 0.5; X_train[:,1:,:]=0
X_test  = trajs_test  - 0.5; X_test[:,1:,:]=0
y_train = trajs_train
y_test  = trajs_test



model.fit(X_train, y_train, nb_epoch=5, batch_size=10)
model.fit(X_train, y_train, nb_epoch=2, batch_size=5)
model.fit(X_train, y_train, nb_epoch=1, batch_size=1)


print "Elapsed time: %0.3f" % (time.time() - start_time)
