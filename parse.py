import pandas as pd
df = pd.read_csv('test2.txt', comment='#', sep=' *\| *', engine='python')
df['Effs2'] = df.Effs.apply(lambda s: np.fromstring(s[1:-1], sep=' '))

