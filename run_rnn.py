import pandas as pd
import numpy as np
from utils import timeIt
import trainrnn
import logging    
from loaddata import get_data

logging.getLogger('keras').setLevel(logging.INFO)

def get_model(num_timesteps, num_input_vars, num_output_vars, p_params=None):
    params = dict(archtype='rnn',optimizer='rmsprop',discount=0.9, hidden_layer_dims=[])

    if p_params:
        params.update( p_params )

    #params.update( dict(macro_dims=500, archtype='RNN',hidden_activation='leakyrelu') ) 
    #params.update( dict(macro_dims=500, hidden_activation='leakyrelu') ) # gets better than mean perfornace on ieee300
    #params.update( dict(macro_dims=trajs_trn.shape[2], archtype='RNNIdentity', hidden_activation='leakyrelu') ) # gets better than mean perfornace on ieee300
    #params.update( dict(macro_dims=trajs_trn.shape[2], archtype='RNNIdentity', hidden_activation='linear') ) # gets better than mean perfornace on ieee300
    #params.update( dict(macro_dims=500, hidden_activation='srelu') )# , hidden_layer_dims=[500,]) 


    model_name = "models/" + "_".join(["%s"%x[1] for x in sorted(params.items()) if x[1]]) + \
                 "-" + "-".join(map(str, [num_timesteps, num_input_vars, num_output_vars]))

    if False:
        # elsewhere...
        from keras.models import model_from_json
        with timeIt("Loading model object"):
            model = model_from_json(open('%s.json'%model_name).read())
            model.load_weights('%s_weights.h5' % model_name)
    else:
        with timeIt("Creating model object"):
            model = trainrnn.get_rnn_model(num_timesteps, num_input_vars, num_output_vars=num_output_vars, output_type='real', **params)
    
    model.model_name = model_name
    return model

"""
def save_model(model):
    print
    print "Saving", model.model_name
    open('%s.json'%model.model_name, 'w').write(model.to_json())
    model.save_weights('%s_weights.h5' % model.model_name, overwrite=True)    
"""

if __name__ == "__main__":
    USE_TOP    = 1e8
    batch_size = 10
    trajs, trajs_trn, observable = get_data()
    from keras.callbacks import ModelCheckpoint
    
    NUM_EPOCHS=1000
    dims = [3,10,50,100,300,500,700,1000]
    regs = [0,]
    #archs = ['rnn','rnn2','rnn_identity','rnn_init_time','rnn1']
    #archs = ['rnn3',]
    #archs = ['rnn4',]
    #archs = ['rnn4_stacked',]
    archs = ['rnn2',]
    #archs = ['rnn2',]; dims = [2,3,5,10]
    #archs = ['rnn2',]; dims = [50,100]; NUM_EPOCHS = 1000
    #archs = ['rnn2',]; dims = [2,3,5,10]
    #archs = ['rnn2_invstacked',]; dims = [3]; NUM_EPOCHS = 1000
    #regs = [0, 1e-3, 1e-2, 1e-1, 0.5, 1]; dims = [100,]
    #archs = ['rnn2_invstacked',]; NUM_EPOCHS = 100; regs = [1e-5,1e-4,1e-3]; dims = [100,] # 1e-3, 1e-1, 1, 0]; 
    #archs = ['rnn2_invstacked',]; NUM_EPOCHS = 500; dims=[500,] # regs = [1e-5,1e-4,1e-3]; dims = [100,] # 1e-3, 1e-1, 1, 0]; 

    
    #archs = ['rnn2',]; dims=[500,] ; regs = [5e-7,1e-6,2e-6,3e-6,4e-6,5e-6,1e-5] # ; dims = [100,] # 1e-3, 1e-1, 1, 0]; 
    #archs = ['rnn2',]; dims=[500,]; observable = observable[:,:,280][:,:,None]
    archs = ['rnn2',]; dims=[500,]; observable = observable[:,:,[2, 6, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 118, 128, 129, 135, 136, 141, 145, 146, 147, 148, 149, 159, 162, 164, 166, 262]]
    
    #regs = [0, 1e-3, 1e-2, 1e-1, 0.5, 1]
    for r in regs:
        for d in dims:
            for a in archs:
                params = dict(macro_dims=d, archtype=a,hidden_activation='tanh', macro_activation='tanh', regularization=r)
                #params['optimizer']='adam'
                print
                print
                print "**********************"
                print "DOING arch=%s, dims=%d, reg=%g" % (a, d, r)
                model  = get_model(trajs_trn.shape[1], trajs_trn.shape[2], observable.shape[2], params)
                with open('%s.json'%model.model_name, 'w') as f:
                    f.write(model.to_json())
                savecallback = ModelCheckpoint('%s_weights.h5' % model.model_name, 
                                                               monitor='loss', 
                                                               verbose=1,
                                                               save_best_only=True)
                # model.fit(trajs_trn[:USE_TOP], observable[:USE_TOP], nb_epoch=NUM_EPOCHS-20, batch_size=batch_size, 
                #          validation_split=0.1, verbose=2)
                # # save best of last
                model.fit(trajs_trn[:USE_TOP], observable[:USE_TOP], nb_epoch=NUM_EPOCHS, batch_size=batch_size, 
                          validation_split=0.1, verbose=2, callbacks=[savecallback,])

                print 
                print
            


