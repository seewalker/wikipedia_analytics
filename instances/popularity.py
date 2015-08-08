import psycopg2
import pandas as pd
import numpy as np
import math

def popSample (num_samples=7,partition='quantile',thresh=80):
   conn = psycopg2.connect(dbname="wiki", user="shalom",
                           host="localhost", password="")
   sample = pd.read_sql('''SELECT sum(n), title FROM wikiThresh
                           WHERE random() < {}
                           GROUP BY title
                           HAVING sum(n) > {}
                           ORDER BY sum(n) DESC
                        '''.format(0.00001,thresh), conn)
   if partition == 'quantile':
       return np.array_split(sample,num_samples)
   elif partition == 'even':
       view_block = math.ceil(np.sum(sample['sum'].values) / num_samples)
       count, parts = 0, [ set() for i in range(num_samples)]
       for (i, (views,agged_title)) in sample.iterrows(): #this relies on the sorted order
           count += views
           parts[math.floor(count / view_block)].add(agged_title)
       return parts
   else:
       print("unimplemented")

evens = popSample(num_samples=5,partition='even')
even_inst = {
   'name' : 'popularity quantiles',
   'experimental' : {
      '1' : evens[0],
      '2' : evens[1],
      '3' : evens[2],
      '4' : evens[3],
      '5' : evens[4]
   },
   'control' : { }
}

quantiles = popSample(num_samples=5,partition='quantile')
quantile_inst = { #do people click towards more obscure or less obscure things?
   'name' : 'popularity quantiles',
   'experimental' : {
      '0 quantile' :  set(quantiles[0]['title']),
      '.2 quantile' : set(quantiles[1]['title']),
      '.4 quantile' : set(quantiles[2]['title']),
      '.6 quantile' : set(quantiles[3]['title']),
      '.8 quantile' : set(quantiles[4]['title']),
   },
   'control' : { }
}

