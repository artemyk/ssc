import os
import pandas as pd
import traceback
import numpy as np

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
