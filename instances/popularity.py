import psycopg2
import pandas as pd
import numpy as np

def popSample (num_samples=7,partition='quantile',thresh=80):
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

quantiles = popSample(num_samples=5,partition='quantile')
quantile_inst = { #do people click towards more obscure or less obscure things?
   'name' : 'popularity flow',
   'experimental' : {
      '0 quantile' :  set(quantile[0]['title']),
      '20 quantile' : set(quantile[1]['title']),
      '40 quantile' : set(quantile[2]['title']),
      '60 quantile' : set(quantile[3]['title']),
      '80 quantile' : set(quantile[4]['title']),
   },
   'control' : { }
}
