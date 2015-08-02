from __future__ import division
import psycopg2
import pandas as pd
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import seaborn

# Really, what is the best way for me to get scalar field values?
#   if I use a panel, I will have one extra thing to index by, which can be title. This will allow
#   me to use title as a categorical label in seaborn plotting commands, which is what I want to do.
#   if I go this route and it really has to be uniformly tabular (all titles available to all categories),
#   I'll have to have missing values. I should ask on stack overflow whether I should do this or whether
#   there is another recommended approach.

class WikiProblem:
    '''
    An instance is of the form:
       { 'experimental' : { category0 : set( <insert_sample> ), ... categoryN : set( <insert_sample> )}
         'control' : { category0 : set( <insert_sample> ), ... categoryM : set( <insert_sample> )} }
    All experimental categories which are not random are thought to be related somehow.
    All control categories are intended catch an illusury correlation.
    Notes on the Data:
       'wikipedia' as an engine means a use of the wikipedia search bar.
       The rationale for having a threshold is that low-view pages will have noisy proportions
       (e.g. wow! three times as many views by links)
    Notes on the Code:
       Execution of EngineProblem objects anticipates the database to be static as its executing.
    '''
    def __init__(self, instance, checkSamples=True, fromData=None):
        if fromData:
            self = pickle.load(fromData)
        else:
           self.instance, self.checkSamples = instance, checkSamples
           #this is the standard ordering of the categories.
           self.exp_categories = {category for category in self.instance['experimental']}
           self.control_categories = {category for category in self.instance['control']} | {'random'}
           self.categories = self.exp_categories | self.control_categories
           self.engines = {'google','bing','yahoo','wikipedia'}
           # The things I'm interested in
           self.attributes = {'prop-' + eng for eng in self.engines} | {'engines/links'}
           self.random_size = 800
           self.prepare_instance()
           # This two are necessary for making the data properly tabular.
           #self.exp_titles = functools.reduce(lambda x,y: x | y, self.instance['experimental'].values())
           #self.control_titles = functools.reduce(lambda x,y: x | y, self.instance['control'].values())
           exps = { category : pd.DataFrame(data=np.ndarray((len(self.instance['experimental'][category]),len(self.attributes))),
                                            index=self.instance['experimental'][category],
                                            columns=self.attributes) for category in self.exp_categories}
           ctrls = { category : pd.DataFrame(data=np.ndarray((len(self.instance['control'][category]),len(self.attributes))),
                                            index=self.instance['control'][category],
                                            columns=self.attributes) for category in self.control_categories}
           self.experimental = pd.Panel(exps)
           self.control = pd.Panel(ctrls)
           self.churn()
           self.stats()
    def __del__ (self):
        # save states and free the resources when the object gets garbage collected.
        self.cursor.close()
        self.conn.close()
        del(self.conn)
        del(self.cursor) # delete these because they may carry password info that I don't want to pickle.
        # is this working?
        pickle.dump(self, open('results/{}'.format(self.instance['name']), 'wb'))
    def prepare_instance(self):
        '''
        Setup database connection and toss out titles in the samples that are not in the database.
        Sets colors corresponding to categories, sets self.dbsize.
        '''
        print("Preparing problem instance.")
        self.conn = psycopg2.connect(dbname="wiki", user="shalom", host="localhost", password="")
        self.cursor = self.conn.cursor()
        #self.cursor.execute("SELECT count(*) FROM wikiThresh")
        #self.dbsize = self.cursor.fetchall()[0][0]
        self.dbsize = 16670685  # the above takes awhile, so using this while in development.
        if self.checkSamples:
           for t in ('experimental','control'):
               for category in self.instance[t].keys():
                   removals = []
                   for title in self.instance[t][category]:
                       try:
                           self.cursor.execute('''SELECT DISTINCT title FROM wikiThresh
                                                  WHERE title = '{}' '''.format(title))
                           nonexistentTitle = self.cursor.fetchall() == []
                           if (nonexistentTitle):
                               print("{} does not exist".format(title))
                               removals.append(title)
                       except psycopg2.DatabaseError as e:
                           print("Failed to read table")
                           return None
                   for title in removals: self.instance[t][category].remove(title)
        assert(self.random_size < self.dbsize) #otherwise, next line will fail. This should not be true anyway.
        self.instance['control']['random'] = self.randomSample()
    def churn(self):
        print("Filling tables with query results")
        def store(category,title,experimental):
            props = self.engineByEngine(self.byTitle(title))
            if experimental: table = self.experimental
            else: table = self.control
            for eng in self.engines:
                table.ix['prop-' + eng,category][title] = props[eng]
            table.ix['engines/links',category][title] = self.enginesVsLinks(self.byTitle(title))
        for t in ('experimental','control'):
            for category in self.instance[t]:
                for title in self.instance[t][category]:
                    store(category,title,t == 'experimental')
        self.experimental.to_latex(buf='results/{}.txt'.format(self.instance['name']))
    def agg(self):
        pass
    def stats(self):
        '''
        What am I interested in?
            overall usage for each engine, aggregated over all titles in a category.
            overall usage of engines versus links, aggregated over all titles in a category.
            overall usage for each engine, aggregated over all categories in the experimental group.
            overall usage of engines versus links, aggregated over all categories in the experimental group.
        What single numbers would say interesting things?
        '''
        pass
    def control_view(self):
        " "
        return self.table[list(self.control_categories)]
    def cluster(self,n=2):
        '''
        Many instances may have in mind a binary expectation. This shows whether the data supports that estimation.
        At the end, counts of what are in each cluster are done. The other number to try clustering about is the
        number of categories in the experiment.
        '''
        pass
    def discovery(self,categories=None):
        ''' The idea is, after discovery, 
        '''
        # Now, I've kind of thought through this a bit. I can play with this interactively,
        # and add stuff here as I go.
        print("Beginning the plotting process.")
        #I can declare my subplots manually matplotlib style, and tell seaborn to target them via the 'ax' keyword argument to the plotting functions.
        #however, the architecture I might want to switch to should maybe have
        def flush(pltfn):
            def wrap():
                pltfn()
                plt.savefig("results/{}".format(self.instance['name']))
                plt.show()
            return wrap
        # #There is a notion of 'figure' for each of these.
        # @flush
        # def plot_density():
        #     fig, (ax1,ax2,ax3) = plt.subplots(3)
        #     sns.stripplot(data=, x="category" , y=, ax=ax1)
        #     sns.violinplot(, ax=ax2)
        #     sns.boxplot(, ax=ax3)
        # @flush
        # def plot_each_cat(): 
        #     # What about a pairplot for each experimental/control pair? What a pair plot does is 
        #     # I need to prepare special dataframes here which can be inserted what's expected into these plotting functions.
        #     for cat in categories:
        #         engine_dat = 
                # engines = sns.FacetGrid(engine_dat, row="engine", col="control")
                # # can I map a jointplot? 
                # engines.map(sns.distplot( ))
            # link_dat = 
            # links = sns.FacetGrid(link_dat, row="category", col="control")
            # links.map(sns.distplot( ))
        # @flush
        # def plot_agg():
            # engine_dat  = 
            # engines = sns.FacetGrid 
    def byTitle(self,title):
        ''' The reason to have this function instead of doing a SQL GROUPBY is that I'm looking for a sample of specific titles, and I'm not
            going to hardcode thousands into a query string.
        '''
        title = title.replace("'","''") # this is the postgres escape code to allow titles with single quotes in them.
        try:
            return pd.read_sql(sql='''SELECT n, referer FROM wikiThresh
                                      WHERE title = '{}'
                                   '''.format(title), con=self.conn)
        except psycopg2.DatabaseError as e:
            print("Failed to read table")
            return None
    def engineByEngine(self,df):
        "proportion of engine popularity in title compared to google by search engine.  if no searches, 0 proportion for all engines."
        def extract(engine):
            vals = df[df['referer'] == 'other-{}'.format(engine)]['n'].values
            if vals.size == 0: return 0
            else: return vals[0]
        counts = {engine : extract(engine) for engine in self.engines}
        denom = sum(counts.values())
        if denom == 0: denom = 1 #to avoid division by zero answers, which maintaining zero of everything
        return { k : counts[k] / denom for k in counts}
    def enginesVsLinks(self,df):
        '''proportion of internal link referer versus search engine referer.
           decision - assume 'other' is mostly links'''
        engines = map(lambda x: 'other-' + x, self.engines)
        engine_vals = df.loc[ df['referer'].isin(engines)]['n'].values
        link_vals = df.loc[~df['referer'].isin(engines)]['n'].values
        if engine_vals.size == 0: engine_vals = [0]
        if link_vals.size == 0  : return "No internal links."
        else: return np.sum(engine_vals) / np.sum(link_vals)
    def randomSample(self):
        '''The titles are not guarenteed to be distinct because of difficulty in combining SQL's order by and distinct ideas, but
         given the size this can scale to, that seems like a small concern in proper sampling. '''
        try:
            # This selects random rows from the database only for titles where the 'n' value for search engine referer rows are all non-zero.
            # It turns out we don't get 0 values, we get simply no entry for other-bing.
            rs = pd.read_sql(sql='''SELECT title FROM wikiThresh w0
                                    ORDER BY RANDOM()
                                    LIMIT {}'''.format(self.random_size), con=self.conn)
            return set(rs['title'].values)
        except psycopg2.DatabaseError as e:
            print("Failed to read random sample from table: " ++ repr(e))
            return None

# plans
# Rethinking Visualization:
  # - replace the scatter plot with a 'stripplot'.
  # - remove the explicit ggplot styling; use matplotlib instead.
  # - keep the rest of what I have, except maybe using a 'distplot' rather than a histogram would be more consise. This would allow me to avoid the explicit fitting, and to also add rugs if that would be helpful.
  # - add violin plots which show the same thing as the histograms.
  # - play around with different pallettes.
  # - how does one target a subplot with seaborn?

  # questions
  # - maybe there is a higher-level way to do the merge?
  # - what kinds of plots go well with attempts at clustering?

# Rethinking data model:
  # - Currently, there are a few distinct dataframes, some of which have dicts as their cell values. Maybe there ought to be a single frame,
  #   and the things I'm curious about it can be methods of the class which observe that single frame. This single frame can have more
  #   attributes instead of having dict bodies.
  # - An idea - a row per search engine, instead of a dict.
# Improving performance:
  # - Once I have already rethought the data model, I can consider rethinking the SQL queries that will populate it. 
# This raw matplotlib versus seaborn thing is a good match for having branches.

# other helper functions
def most_like(phrase,thresh=0.3):
   conn = psycopg2.connect(dbname="wiki", user="shalom", host="localhost", password="")
   lower, upper = phrase[0].lower(), phrase[0].upper()
   hits = pd.read_sql('''SELECT similarity('{0}',filtered.title), filtered.title
                         FROM (SELECT DISTINCT title FROM wikithresh WHERE (title LIKE '{1}%') OR (title LIKE '{2}%')) filtered
                         WHERE similarity('{0}',filtered.title) > {3}
                         ORDER BY similarity('{0}',filtered.title) DESC
                      '''.format(phrase, lower, upper, thresh), conn)
   print(hits.to_string())

# not verified to work.
def dbshape(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM wikiThresh")
    num_rows = self.cursor.fetchall()[0][0]
    cursor.execute("SELECT count(DISTINCT title) FROM wikithresh")
    num_titles = self.cursor.fetchall()[0][0]
    return (num_rows,num_titles)

def proportion(referer):
    return "(SELECT wIn.n :: float / sum(wOut.n) FROM wikithresh wIn WHERE (wIn.title = wOut.title) AND (wIn.referer = '{}'))".format(referer)
# this is interesting just to see what is most done.
def referer_maximal(takeAmount=100):
   "Retrieves the pages which are most often found via certain referers. Research question - do these have anything in common?"
   conn = psycopg2.connect(dbname="wiki", user="shalom", host="localhost", password="")
   def maxes(referer):
       return pd.read_sql(''' SELECT wOut.title, {0}
                              FROM wikiThresh wOut
                              GROUP BY wOut.title
                              ORDER BY {0} DESC
                              LIMIT {1}
                          '''.format(proportion(referer),takeAmount), conn)
   return {referer : maxes(referer) for referer in ('other-google','other-bing','other-yahoo','other-wikipedia')}

def referer_distribution(slice_proportion=0.01):
   "Shows the distribution of proportion of prevalence of referers."
   conn = psycopg2.connect(dbname="wiki", user="shalom", host="localhost", password="")
   def distr(referer):
        # Does seaborn provide a way to make histograms where you can click on the bins to see what members there are?
        return pd.read_sql('''SELECT {0}
                              FROM wikiThresh wOut
                              WHERE random() < {1}
                              GROUP BY wOut.title
                              ORDER BY {0} DESC
        '''.format(proportion(referer),slice_proportion), conn)
   return {referer : distr(referer) for referer in ('other-google','other-bing','other-yahoo','other-wikipedia')}

#Is there a name for something like a histogram, but which has constant count height and variable 'width'?
#if so, I could use that as an alternative way to pose the popularity instance.
# just plot a histogram with the data reversed; what do I want to do this to?
#       number of net views is in a bin, the width corresponds to views per page boundaries.
#       
# def data_characteristics( ):
#     m = referer_maximal(800)
#     d = referer_distribution( )
#     m.to_string
#     plt.hist
