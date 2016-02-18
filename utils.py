
import time    
from contextlib import contextmanager

@contextmanager  
def timeIt(title=None):
    t1 = time.clock()
    yield
    t2 = time.clock()
    if title is not None: print '%s:'%title,
    print '%0.2fs elapsed' % (t2-t1)
