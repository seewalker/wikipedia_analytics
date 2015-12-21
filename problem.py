from __future__ import division
import math
import pickle
import subprocess
import itertools
import numpy as np
import scipy
import psycopg2
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn import gaussian_process
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh import mpl
from bokeh.plotting import output_file, show

# some global variables
ENGINES = {'google','bing','yahoo','wikipedia','facebook','twitter'}

# maybe this should be labeled 'wikiForward'
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
    def __init__(self, instance, checkSamples=True, fromData=None, selection=ENGINES):
        if fromData:
            self = pickle.load(fromData)
        else:
           self.instance, self.checkSamples = instance, checkSamples
           #this is the standard ordering of the categories.
           self.exp_categories = {category for category in self.instance['experimental']}
           self.control_categories = {category for category in self.instance['control']} | {'random'}
           self.categories = self.exp_categories | self.control_categories
           self.engines = selection - {'engines/links'}
           # The things I'm interested in
           self.attributes = selection
           self.random_size = 500
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
           subprocess.call(["rm","-rf","results/{}".format(self.instance['name'])])
           subprocess.call(["mkdir","results/{}".format(self.instance['name'])])
           self.experimental = pd.Panel(exps)
           self.control = pd.Panel(ctrls)
           self.churn()
    def __del__ (self):
        # save states and free the resources when the object gets garbage collected.
        self.cursor.close()
        self.conn.close()
        name = self.instance['name']
        #self.experimental.to_hdf('results/{}/experimental.hdf'.format(name))
        self.experimental.to_pickle('results/{}/experimental.pickel'.format(name))
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
                table[category][eng][title] = props[eng]
            if 'engines/links' in self.attributes:
                table.ix[category]['engines/links'][title] = self.enginesVsLinks(self.byTitle(title))
        for t in ('experimental','control'):
            for category in self.instance[t]:
                for title in self.instance[t][category]:
                    store(category,title,t == 'experimental')
    def stats(self):
        '''
        What am I interested in?
            overall usage for each engine, aggregated over all titles in a category.
            overall usage of engines versus links, aggregated over all titles in a category.
            overall usage for each engine, aggregated over all categories in the experimental group.
            overall usage of engines versus links, aggregated over all categories in the experimental group.
        What single numbers would say interesting things?
        '''
        #pd.set_option('precision',4) #effects the output precision, so that more can be displayed horizontally.
            #weirdly, this didn't work.
        cors = {}
        for (name,df) in self.experimental.iteritems():
            print("{}\n{}".format(name,str(df.describe())))
            corrmat = pd.DataFrame(columns=self.attributes, index=[name for (name,vals) in self.control.iteritems()])
            pd.set_option('precision',3)
            print("The first item in these tuples represents the pearson correlation coefficient.\n")
            print("The second item in these tuples represents the p-value.\n")
            for (featurename,feature) in df.iteritems():
                for (cname,ctrl) in self.control.iteritems():
                    size = min(len(feature.dropna()),len(ctrl[featurename].dropna()))
                    featurecol, ctrlcol = (feature.dropna())[0:size], (ctrl[featurename].dropna())[0:size]
                    # does this pearson coefficient assume minus-mean form? no, what I'm writing here is valid.
                    corrmat[featurename][cname] = scipy.stats.pearsonr(featurecol.values,ctrlcol.values)
            cors[name] = corrmat
            print(corrmat)
        self.correlations = pd.Panel(cors)
        # here, get the mean and variance of each column in random.
        # Here can go some statistical tests, e.g. Kruskal-Wallis.
    def popularityOf(self):
        '''
        Visualizes popularity of pages in experimental categories in general.
        '''
        exps = pd.DataFrame(columns=self.exp_categories)
        for ecat in self.instance['experimental']:
            for title in self.instance['experimental'][ecat]:
                title = title.replace("'","''") # this is the postgres escape code to allow titles with single quotes in them.
                ns = pd.read_sql("SELECT sum(n) FROM wikithresh WHERE title = '{}' GROUP BY title".format(title), con=self.conn).iloc[0][0]
                exps[ecat][title] = int(ns)
        rands = pd.Series()
        for title in self.instance['control']['random']:
            title = title.replace("'","''") # this is the postgres escape code to allow titles with single quotes in them.
            rands[title] = pd.read_sql("SELECT sum(n) FROM wikithresh WHERE title = '{}' GROUP BY title".format(title), con=self.conn).iloc[0][0]
        fig, axes = plt.subplots(nrows=len(self.exp_categories))
        if len(self.exp_categories) == 1: axes = [axes]
        for (i,ecat) in enumerate(self.exp_categories):
            sns.distplot(exps[ecat].apply(lambda x: math.log(x)), ax=axes[i])
            sns.distplot(rands.apply(lambda x: math.log(x)), ax=axes[i], color='r')
        plt.show()
    def discovery(self,categories=None):
        ''' The idea is, after discovery, I will do more specific plots to hone in on what this suggests is interesting data.
        '''
        self.stats()
        figuresize = (36,16) #units - inches
        aes = {'other-google' : 'green',
               'other-wikipedia' : 'black',
               'other-facebook' : 'white',
               'other-twitter' : 'blue',
               'other-yahoo' : 'yellow',
               'other-bing' : 'red',
               'engines/links' : 'purple'}
        def flush(plottype):
            def inner(plotfn):
                def wrap():
                    plotfn()
                    # add a legend to the figure.
                    plt.savefig("results/{}/{}".format(self.instance['name'], plottype))
                    #output_file("results/{}/{}.html".format(self.instance['name'], plottype))
                    plt.show()
                    #show(mpl.to_bokeh())
                return wrap
            return inner
         #There is a notion of 'figure' for each of these.
        print("Plotting density")
        # removing the random column and doing a 'split' will go a long way with making the violins bigger.
        @flush("raw_density")
        def raw_density():
            "Here, I am working with so-called 'wide-form' data."
            fig, axes = plt.subplots(nrows=len(self.exp_categories), ncols=len(self.control_categories)+1, sharey=True)
            fig.set_size_inches(figuresize)
            fig.suptitle("Raw Proportion Values")
            sns.set_style("ticks",{"ytick.major.style" : 0.1, "ytick.minor.style" : 0.025})
            # Things I need - better styling.
            if len(self.exp_categories) == 1:
                axes = [axes]
            for (i,ecat) in enumerate(self.exp_categories):
                axes[i][0].set_title(ecat)
                sns.violinplot(data=self.experimental[ecat], ax=axes[i][0], width=1.0, inner="box", scale="area")
                for (j,ccat) in enumerate(self.control_categories):
                    axes[i][j+1].set_title(ccat)
                    sns.violinplot(data=self.control[ccat], ax=axes[i][j+1], width=1.0, inner="box", scale="area")
        raw_density()
        @flush("deltaRandom_density")
        def dRandom_density():
            fig, axes = plt.subplots(nrows=len(self.exp_categories), ncols=len(self.control_categories)+1, sharey=True)
            fig.set_size_inches(figuresize)
            fig.suptitle("Differences Between Categories And Random Sampling")
            sns.set_style("ticks",{"ytick.major.style" : 0.04, "ytick.minor.style" : 0.01})
            if len(self.exp_categories) == 1:
                axes = [axes]
            expectation = self.control['random'].apply(np.mean)
            for (i,ecat) in enumerate(self.exp_categories):
                axes[i][0].set_title(ecat)
                delta = self.experimental[ecat].apply(lambda row: row - expectation, axis=1)
                sns.violinplot(data=delta, ax=axes[i][0], width=1.0, inner="box", scale="area")
                for (j,ccat) in enumerate(self.control_categories):
                    axes[i][j+1].set_title(ccat)
                    delta = self.control[ccat].apply(lambda row: row - expectation, axis=1)
                    sns.violinplot(data=delta, ax=axes[i][j+1], width=1.0, inner="box", scale="area")
        dRandom_density()
        # What am I interested about with the combined thing?
        @flush("combined_density")
        def combined_density():
            fig, axes = plt.subplots(nrows=len(self.exp_categories) + 1)
            fig.set_size_inches(figuresize)
            fig.suptitle("Differences Between Categories and Combined Experimental Categories")
            if len(self.exp_categories) == 1:
                combined = self.experimental[list(self.exp_categories)[0]]
            else:
                cats = list(self.exp_categories)
                combined = self.experimental[cats[0]].append([self.experimental[cat] for cat in cats[1:]])
            combined_expectation = combined.apply(np.mean)
            random_expectation = self.control['random'].apply(np.mean)
            combined_minus_rand = combined.apply(lambda row: row - random_expectation, axis=1)
            sns.violinplot(data=combined_minus_rand, ax=axes[0], width=1.0, inner="box", scale="area")
            for (i,ecat) in enumerate(self.exp_categories):
                axes[i + 1].set_title(ecat)
                delta = self.experimental[ecat].apply(lambda row: row - combined_expectation, axis=1)
                sns.violinplot(data=delta, ax=axes[i + 1], width=1.0, inner="box", scale="area")
        combined_density()
        @flush("split_random")
        def split_random():
            # here, I'll transform the data so it is grouped by attribute, and the plots are split compared to random.
            pass
        split_random()
        # some bokeh plots go here. what else do I want to highlight? things about the correlation? things about the clustering?
        # something like http://bokeh.pydata.org/en/latest/docs/gallery/iris.html for clustering.
        # what correlates with what could be a heatmap.
        @flush("correlation")
        def corr_heat():
            # can I do a panel-based row of heatmaps with a single line of seaborn code? if not ...
            fig, axes = plt.subplots(nrows=len(self.experimental))
            fig.suptitle("Correlations")
            if len(self.experimental) == 1: axes = [axes]
            i = 0
            for (ecat, df) in self.correlations.iteritems():
                axes[i].set_title(ecat)
                # for now, just plotting correlation coefficients
                sns.heatmap(data=df.applymap(lambda x:x[0]),ax=axes[i])
                i += 1
        corr_heat()
        # holy shit, there is a function: `show(mpl.to_bokeh)` that will allow me the zoom thing, perfect for my discovery thing.
    def ml(self):
        "What happens when various machine learning methods are applied?"
        cluster()
        separate()
    def cluster(self,cats,dims,n):
        assert(set(cats).issubset(self.exp_categories) and (len(dims) > 1) and set(dims).issubset(self.attributes))
        self.km = sklearn.cluster.KMeans(n_clusters=n)
        if len(cats) > 1:
            df = self.experimental[cats[0]].append([self.experimental[cat] for cat in cats[1:]]).dropna()
        else:
            df = self.experimental[cats[0]]
        self.km.fit(df[list(dims)])
    def separate(self):
        dfs, y = [], []
        cats = list(self.exp_categories)
        for cat in cats:
            X = self.experimental[cat].dropna().copy()
            dfs.append(X)
            y += [cat for i in range(len(X.major_axis))]
        self.svm = SVC()
        self.svm.fit(dfs[0].append(dfs[1:]), y)
    # todo - add gaussian processes and a probabilistic output classifiers.
    def predict(self):
        pass
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

# this is interesting just to see what is most done.
def referer_maximal(takeAmount=100):
   "Retrieves the pages which are most often found via certain referers. Research question - do these have anything in common?"
   conn = psycopg2.connect(dbname="wiki", user="shalom", host="localhost", password="")
   # does the notion of proportion require a subquery?
   def proportion(referer):
       return "((SELECT wIn.n FROM sample wIn WHERE (wIn.title = wOut.title) AND (wIn.referer = '{}')) / (sum(wOut.n) :: Float))".format(referer)
   def maxes(referer):
       print("doing on of em.")
       return pd.read_sql(''' SELECT wOut.title, {0}
                              FROM sample wOut
                              WHERE (SELECT wIn.n FROM sample wIn WHERE (wIn.title = wOut.title) AND (wIn.referer = '{2}')) IS NOT NULL
                              GROUP BY wOut.title
                              ORDER BY {0} DESC
                              LIMIT {1}
                          '''.format(proportion(referer),takeAmount,referer), conn)
   xs = maxes('other-google')
   return {referer : maxes(referer) for referer in map(lambda x: 'other-' + x, ENGINES)}

def pageview_distribution(proportion=0.08,nbins=100):
    conn = psycopg2.connect(dbname="wiki", user="shalom", host="localhost", password="")
    distr = pd.read_sql('''SELECT title, sum(n)
                           FROM wikiThresh
                           WHERE random() < {}
                           GROUP BY title
                           ORDER BY sum(n) DESC'''.format(proportion), conn)
    fig, (ax1,ax2) = plt.subplots(nrows=2)
    sns.distplot(distr['sum'], bins=nbins, ax=ax1)
    sns.distplot(distr['sum'].apply(math.log), bins=nbins, ax=ax2)
    plt.show()

def data_characteristics( ):
    maxes = referer_maximal(30)
    for (referer,df) in maxes.items():
        print(referer)
        print(df)
    pageview_distribution()
    random_inst = {
            'name' : 'random_sample',
            'experimental' : {},
            'control' : {}}
    rp = WikiProblem(random_inst)
    print(rp.control['random'].describe())
