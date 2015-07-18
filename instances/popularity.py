import psycopg2
import pandas as pd
import numpy as np

def popSample (num_samples=7,partition='quantile'thresh=80):
   conn = psycopg2.connect(dbname="wiki", user="shalom",
                           host="localhost", password="")
   sample = pd.read_sql('''SELECT sum(n), title FROM wikiThresh
                           WHERE random() < {}
                           GROUP BY title
                           HAVING sum(n) > {}
                           ORDER BY sum(n) DESC
                        '''.format(0.001,thresh), conn)
   if partition == 'quantile':
       return np.array_split(sample,num_samples)
   elif partition == 'even':
       return
   else:
       print("unimplemented")

quantiles = popSample(num_samples=7,partition='quantile')
quantile_inst = { #do people click towards more obscure or less obscure things?
   'name' : 'popularity flow',
   'experimental' : {
      '0 quantile' :  set(sample[0]['title']),
      '20 quantile' : set(sample[1]['title']),
      '40 quantile' : set(sample[2]['title']),
      '60 quantile' : set(sample[3]['title']),
      '80 quantile' : set(sample[4]['title']),
   },
   'control' : { }
}
evens = popSample(num_samples=7,partition='quantile')

