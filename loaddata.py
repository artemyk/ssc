import os
import pandas as pd
import traceback
import numpy as np
from utils import timeIt


def get_data(DIRNAME='ieee300'):
    #DIRNAME = 'alpha1.0r0.0'
    TOPNUM  = None
    with timeIt():
        df = get_df(DIRNAME, TOPNUM)
        df['PertId'] = df.PertId.apply(lambda x: tuple(sorted(str(x).split(','))))

    def split_mx(df, unique_perts=False):
        done_perts = set()
        clocs = None
        pertDF = df[df.PertId!=-1]

        def emit(clocs):
            cPerts = pertDF.PertId.iloc[clocs[0]] 
            if unique_perts and cPerts in done_perts:
                return None
            done_perts.add(cPerts)
            return np.vstack(pertDF.Effs2.iloc[clocs]), np.vstack(pertDF.Eff.iloc[clocs])
        for ndx, t in enumerate(pertDF.t):
            if t == 0:
                if clocs is not None: 
                    if len(clocs) == 15:
                        d = emit(clocs)
                        if d is not None:
                            yield d
                clocs = []
            clocs.append(ndx)
        if len(clocs) == 15:
            d = emit(clocs)
            if d is not None:
                yield d

    with timeIt("Loading trajectories"):
        mxs = zip(*split_mx(df,unique_perts=True))
        trajs = np.stack(mxs[0])
        #trajs = trajs[0:1000]
        trajs_trn = trajs.copy()
        trajs_trn[:,1:,:] = 0.0

        if False:
            num_mostvaried = 1
            #mostvaried=np.argsort(trajs[:,-1,:].var(axis=0))[-num_mostvaried:]
            #observable = trajs[:,:,mostvaried]
            observable = trajs - trajs.mean(axis=0)[None,:,:]
            print trajs.shape, observable.shape
        observable = trajs
        #observable = np.stack(mxs[1])*1000
        v = ((observable - observable.mean(axis=0)[None,:,:])**2)
        print "Mean error for avg           : %0.7f" % v.mean()
        print "Mean error for avg (last10%%) : %0.7f" % v[int(len(v)*.9):].mean()
        print "Mean error for avg (frst10%%) : %0.7f" % v[:int(len(v)*.1)].mean()
    return trajs, trajs_trn, observable

def get_df(DIRNAME, TOPNUM):
    PKL_FILE = DIRNAME + ('top%d'%TOPNUM if TOPNUM is not None else '') + '_v2.pkl'
    if not os.path.exists(PKL_FILE):
        print "Creating pickle file %s" % PKL_FILE
        from io import StringIO
        #import zipfile
        #
        #def get_dfs(zipfilename):
        #    f = zipfile.ZipFile(zipfilename)
        #    for zname in f.namelist():
        #                   with f.open(zname) as zf:


        def get_dfs(dirname, usetop=10000):
            for zname in sorted(os.listdir(dirname))[:usetop]:
                print "Doing", zname
                with open(dirname + '/' + zname) as zf:
                    lines = zf.readlines()
                    if "job killed" in lines[-1]:
                        lines = lines[:-1]
                    df = pd.read_csv(StringIO(u"\n".join(lines)), comment='#', sep=' *\| *', engine='python')
                    #df = pd.read_csv(zf, comment='#', sep='|')
                    try:
                        df['Effs2'] = df.Effs.apply(lambda s: np.fromstring(s[1:-1], sep=' ').astype('float32'))
                        yield df[['PertId','t','Eff','Effs2','RunT']]
                    except:
                        print "Error encountered!"
                        traceback.print_exc()

        df = pd.concat(get_dfs('../ssc-data/'+DIRNAME, TOPNUM))
        df.to_pickle(PKL_FILE)
    else:
        print "Reading pickle file %s" % PKL_FILE
        df = pd.read_pickle(PKL_FILE)

    return df
