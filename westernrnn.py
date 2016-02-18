import pandas as pd
import numpy as np

df = pd.read_pickle('out1.pkl')
def split_mx(df):
    clocs = None
    for ndx, t in enumerate(df[df.PertId!=-1].t):
        if t == 0:
            if clocs is not None: 
                if len(clocs) == 15:
                    yield np.vstack(df.Effs2.iloc[clocs])
            clocs = []
        clocs.append(ndx)
    if len(clocs) == 15:
        yield np.vstack(df.Effs2.iloc[clocs])
    
trajs = np.stack(split_mx(df))


import trainrnn
import logging    
logging.getLogger('keras').setLevel(logging.INFO)
hidden_dims = 350

model = trainrnn.get_rnn_model(dshape=trajs.shape, hidden_dims=hidden_dims, output_type='real')


# model.fit(trajs, trajs, nb_epoch=100, batch_size=10, validation_split=0.1, show_accuracy=True, verbose=1)
#model.fit(trajs, trajs, nb_epoch=10, batch_size=5, validation_split=0.1, show_accuracy=True, verbose=1)
model.fit(trajs, trajs, nb_epoch=100, batch_size=1, validation_split=0.1, show_accuracy=True, verbose=1)

