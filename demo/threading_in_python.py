
# coding: utf-8

# In[4]:


#! usr/bin/env python3
from multiprocessing.dummy import Pool as ThreadPool
import urllib

urls=[
'http://www.python.org',
'http://www.python.org/about/',
'http://www.onlamp.com/pub/a/python/2003/04/17/metaclasses.html',
'http://www.python.org/doc/',
'http://www.python.org/download/',
'http://www.python.org/getit/',
'http://www.python.org/community/',
'https://wiki.python.org/moin/'
]

pool = ThreadPool(4)

results = pool.map(urllib.request.urlopen,urls)

pool.close()
pool.join()

