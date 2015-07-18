from __future__ import division
import psycopg2
import pandas as pd
import numpy as np
import math
import pickle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import norm

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
           self.exp_categories = [category for category in self.instance['experimental']]
           self.control_categories = [category for category in self.instance['control']] + ['random']
           self.engines = ['google','bing','yahoo','wikipedia']
           self.attributes = ['engine/engines','engines/links'] #the things I know to look at.
           self.category_colors = {}
           self.fit = norm.fit #is there a better type of distribution to fit against?
           self.random_size = 800
           self.result = None
           self.prepare_instance()
           self.churn()
           self.stats()
    def __del__ (self):
        # save states and free the resources when the object gets garbage collected.
        self.cursor.close()
        self.conn.close()
        del(self.conn)   #
        del(self.cursor) # delete these because they may carry password info that I don't want to pickle.
        pickle.dump(self, open('results/{}'.format(self.instance['name']), 'wb'))
    def prepare_instance(self):
        '''
        Setup database connection and toss out titles in the samples that are not in the database.
        Sets colors corresponding to categories, sets self.dbsize.
        '''
        print("Preparing problem instance.")
        self.conn = psycopg2.connect(dbname="wiki", user="shalom", 
                                     host="localhost", password="")
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT count(*) FROM wikiThresh")
        self.dbsize = self.cursor.fetchall()[0][0]
        exp_colors = ['#fdfd96', '#96fdc9', '#fdc996', '#96fdfd', '#fd9696', '#5A6afc', '#fd96c9', '#c996fd'] #these should look like pastel colors
        control_colors = ['#0000ff', '#00ff00', '#00ffff', '#ffff00']  #these should look like almost-primary colors.
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
                   if t == 'experimental': self.category_colors[category] = exp_colors.pop()
                   else: self.category_colors[category] = control_colors.pop()
        self.category_colors['random'] = '#ff0000'
        self.all_categories = self.exp_categories + self.control_categories
    def churn(self):
        print("Filling tables with query results")
        #experimental and control tables ought to be stored separately.
        self.experiment = pd.DataFrame(data=self.array_of_dicts((len(self.attributes), len(self.exp_categories))), columns=self.attributes, index=self.exp_categories)
        self.control = pd.DataFrame(data=self.array_of_dicts((len(self.attributes), len(self.control_categories))), columns=self.attributes, index=self.control_categories)
        for category in self.instance['experimental']:
            for title in self.instance['experimental'][category]:
                self.experiment.ix[category,'engine/engines'][title] = self.engineByEngine(self.byTitle(title))
                self.experiment.ix[category,'engines/links'][title] = self.enginesVsLinks(self.byTitle(title))
        for category in self.instance['control']:
            for title in self.instance['control'][category]: #the instance does not explicitly include random.
                self.control.ix[category,'engine/engines'][title] = self.engineByEngine(self.byTitle(title))
                self.control.ix[category,'engines/links'][title] = self.enginesVsLinks(self.byTitle(title))
        assert(self.random_size < self.dbsize) #otherwise, next line will fail. This should not be true anyway.
        for title in self.randomSample(proportion = self.random_size / self.dbsize)['title'].values:
            self.control.ix['random','engine/engines'][title] = self.engineByEngine(self.byTitle(title))
            self.control.ix['random','engines/links'][title] = self.enginesVsLinks(self.byTitle(title))
        self.experiment.to_latex(buf='results/{}.txt'.format(self.instance['name']))
    def stats(self):
        '''
        Formats of results:

        engine_panel:  foreach engine   
                           labels    values   fit   versus-each
                     cat0  
                     cat1
            //labels and values zip together to associate values with titles.

        This has reduced away information about the categories and only have 'is-experimental' knowledge.
        engine_frame:          label     values     fit   versus-each  experimental?
                     google 
                     wikipedia
                     bing
                     yahoo
            //These are sums of stuff in engine_panel.
        link_frame             value  versus-each experimental?
                     cat0
                     cat1
        link_values, link_fit
        '''
        # What other kinds of attributes would be useful to have? Some quantitative measure of clustering of experimental vs control categories.
        # This is something to return to when I know more about statistics.
        print("Reducing the query tables.")
        attributes = ['labels', 'values', 'fit', 'experimental?']
        self.engine_panel = pd.Panel({engine : pd.DataFrame(index=self.all_categories, columns=attributes) for engine in self.engines})
        for engine in self.engines:
           for category in self.control_categories:
               cell = self.control.ix[category,'engine/engines']
               self.engine_panel[engine].ix[category, 'experimental?'] = False
               self.engine_panel[engine].ix[category, 'labels'], self.engine_panel[engine].ix[category, 'values'] = [], []
               for title in cell.keys():
                   self.engine_panel[engine].ix[category, 'labels'].append(title)
                   self.engine_panel[engine].ix[category, 'values'].append(cell[title][engine]) #this line currently creating problems.
               self.engine_panel[engine].ix[category, 'fit'] = self.fit(self.engine_panel[engine].ix[category, 'values'])
           for category in self.exp_categories:
               cell = self.experiment.ix[category,'engine/engines']
               self.engine_panel[engine].ix[category, 'experimental?'] = True
               self.engine_panel[engine].ix[category, 'labels'], self.engine_panel[engine].ix[category, 'values'] = [], []
               for title in cell.keys():
                   self.engine_panel[engine].ix[category, 'labels'].append(title)
                   self.engine_panel[engine].ix[category, 'values'].append(cell[title][engine])
               self.engine_panel[engine].ix[category, 'fit'] = self.fit(self.engine_panel[engine].ix[category, 'values'])
        self.engine_frame = pd.DataFrame(columns=['experiment-values', 'experiment-fit'], index=self.engines)
        for engine in self.engines:
            filtered = self.engine_panel[engine]
            filtered = filtered[ filtered['experimental?'] == True]
            self.engine_frame.ix[engine,'experiment-values'] = np.concatenate(filtered['values'].values)
            self.engine_frame.ix[engine,'experiment-fit'] = self.fit(self.engine_frame.ix[engine,'experiment-values'])
        self.link_frame = pd.DataFrame(index=self.all_categories, columns=['labels','values','fit', 'experimental?'])
        for category in self.all_categories:
            self.link_frame.ix[category,'values'], self.link_frame.ix[category,'labels'] = [], []
            if category in self.exp_categories:
                self.link_frame.ix[category,'experimental?'] = True
                source = self.experiment.ix[category,'engines/links']
            else:
                self.link_frame.ix[category,'experimental?'] = False
                source = self.control.ix[category,'engines/links']
            for label, value in source.items():
                self.link_frame.ix[category,'labels'].append(label)
                self.link_frame.ix[category,'values'].append(value) 
            self.link_frame.ix[category,'fit'] = self.fit(self.link_frame.ix[category,'values'])
        filtered = self.link_frame[ self.link_frame['experimental?'] == True]
        self.link_experimental_vals = np.concatenate(filtered['values'].values)
        self.link_experimental_fit = self.fit(self.link_experimental_vals)
    def plot(self,categories=None):
        '''
         coherence of engine/engines question
         figure 1 (row per category) (column per engine) : each category corresponds to a color (and a x-line). each sample point corresponds to a sample item.
                                                           are experimental categories a cluster? are the control groups in this cluster?
                                                           these plots should be two-dimensional. each of the values corresponds to a title, so they should be clickable.
                                                           Each x-line has a mean tick and standard deviation bars.
         answers to engine/engines question
         figure 2 (row per category) (column per engine) : histograms of values in a category, overlayed with a histogram for each control.
                                                           plot indications of mean and standard deviation. each histogram overlayed with
                                                           a normal pdf. clicking could display list of values in bins.
         figure 3 (one row for all categories) (column per engine) : like above but all experimental categories merged together.
         
         coherence of engines/links question
         figure 4 : (row per category)                   : like figure 2, but with Engines/Links values. Naturally, no engine-by-engine breakdown is relevant, though.

         figure 5 : (one row for all categories)         : like figure 3.
        '''
        # Once all this plotting works, I should find a way to reduce the redundancy in the code.
        def normal(model):
            # the first field is the mean and the second is the standard deviation.
            xs = np.linspace( model[0] - 2 * model[1], model[0] + 2 * model[1], 100)
            return xs, norm.pdf(xs, loc=model[0], scale=model[1])
        def labelOf_engineCoherence(event):
            #print("{} {} {} {} {}\n".format(event.button, event.x, event.y, event.xdata, event.ydata))
            x_tolerance, y_tolerance = 0.1, 0.005
            def lookup(y):
                if math.ceil(event.xdata) - event.xdata < event.xdata - math.floor(event.xdata): x = math.ceil(event.xdata)
                else: x = math.floor(event.xdata)
                for i, (label, value) in enumerate(zip(self.engine_panel[engine].ix[self.all_categories[x], 'label'],
                                                       self.engine_panel[engine].ix[self.all_categories[x], 'value'])):
                      if max(value, event.ydata) - min(value, event.ydata) < y_tolerance:
                          return label
                return False
            if min(math.ceil(event.xdata) - event.xdata, event.xdata - math.floor(event.xdata)) < x_tolerance: #testing for close enough
                label = lookup(event.ydata)
                if label_index: print(label)
                else: print("lookup failed.")
        print("Beginning the plotting process.")
        #plotting paramters:
        engine_colors = {'google' : 'g', 'bing' : 'b', 'yahoo' : 'y', 'wikipedia' : 'r'}
        tickwidth = 0.15
        screensize = (25,40)
        nbins = 8
        exp_alpha, control_alpha = 0.8, 0.2
        if not categories: categories = self.all_categories
        engine_coherence_fig, engine_coherence_axes = plt.subplots(nrows=1, ncols=len(self.engines), figsize=screensize)
        engine_answers_fig, engine_answers_axes = plt.subplots(nrows=len(self.exp_categories), ncols=len(self.engines), figsize=screensize)
        engine_answer_fig, engine_answer_axes = plt.subplots(nrows=1, ncols=len(self.engines), figsize=screensize)
        cid = engine_coherence_fig.canvas.mpl_connect('button_press_event', labelOf_engineCoherence)
        plt.margins(0.25)
        plt.subplots_adjust(bottom=0.15)
        plt.style.use('ggplot')
        engine_coherence_axes[0].set_ylabel('proportion of all searches')
        for i, engine in enumerate(self.engines):
            engine_coherence_axes[i].set_title(engine)
            engine_coherence_axes[i].set_xticks(range(0,len(self.all_categories)))
            engine_coherence_axes[i].set_xticklabels(self.all_categories, rotation='vertical')
            engine_answer_axes[i].set_title(engine)
            engine_answer_axes[i].set_xlabel('proportion of all searches')
            engine_answer_axes[i].hist(self.engine_frame.ix[engine, 'experiment-values'], bins=nbins, normed=True, facecolor='blue', alpha=exp_alpha)
            engine_answer_axes[i].hist(self.engine_panel[engine].ix['random', 'values'], bins=nbins, normed=True, facecolor='red', alpha=control_alpha)
            xs, pdf = normal(self.engine_frame.ix[engine, 'experiment-fit'])
            engine_answer_axes[i].plot(xs, pdf, color='blue')
            xs, pdf = normal(self.engine_panel[engine].ix['random', 'fit'])
            engine_answer_axes[i].plot(xs, pdf, color='red')
            for j, category in enumerate(self.all_categories):
                 vals = self.engine_panel[engine].ix[category,'values']
                 mean, stddev = self.engine_panel[engine].ix[category,'fit']
                 engine_coherence_axes[i].scatter( len(vals) * [j], vals, color=self.category_colors[category])
                 engine_coherence_axes[i].hlines([mean, mean + stddev, mean - stddev], xmin=(j - tickwidth), xmax=(j + tickwidth), colors = ['green', 'red', 'red'])
            for k, category in enumerate(self.exp_categories):
                 vals = self.engine_panel[engine].ix[category,'values']
                 xs, pdf = normal(self.engine_panel[engine].ix[category,'fit'])
                 #how is this index out of bounds?
                 if len(self.exp_categories) > 1: index = k, i #this conditional necessary because matplotlib uses 2D axes array iff nrows and ncols > 1.
                 else: index = i
                 patches = [mpatches.Patch(color=self.category_colors[cat], label=cat) for cat in [category] + self.control_categories]
                 engine_answers_axes[index].legend(handles=patches, prop={'size' : 7})
                 engine_answers_axes[index].set_title("{} | {}".format(engine, category))
                 engine_answers_axes[index].set_ylabel("Normalized Distribution of Searches")
                 engine_answers_axes[index].set_xlabel("Proportion")
                 engine_answers_axes[index].hist(vals,bins=nbins, facecolor=self.category_colors[category], alpha=exp_alpha, normed=True) 
                 engine_answers_axes[index].plot(xs, pdf, color=self.category_colors[category])
                 for control_cat in self.control_categories:
                     vals = self.engine_panel[engine].ix[control_cat, 'values']
                     xs, pdf = normal(self.engine_panel[engine].ix[control_cat,'fit'])
                     engine_answers_axes[index].hist(vals,bins=nbins, facecolor=self.category_colors[control_cat], alpha=control_alpha, normed=True) #lower alpha value 
                     engine_answers_axes[index].plot(xs, pdf, color=self.category_colors[control_cat])
        link_answers_fig, link_answers_ax = plt.subplots(nrows=len(self.exp_categories),ncols=1, figsize=screensize)
        for j, category in enumerate(self.exp_categories):
           if len(self.exp_categories) > 1: ax = link_answers_ax[j]
           else: ax = link_answers_ax
           ax.set_title(category)
           ax.set_xlabel("Proportion of engine referers to link referers")
           ax.hist(self.link_frame.ix[category, 'values'], normed=True, alpha=exp_alpha, facecolor=self.category_colors[category])
           xs, pdf = normal(self.link_frame.ix[category, 'fit'])
           ax.plot(xs, pdf, color=self.category_colors[category])
           for category in self.control_categories:
               ax.hist(self.link_frame.ix[category,'values'], normed=True, alpha=control_alpha, facecolor=self.category_colors[category])
               xs, pdf = normal(self.link_frame.ix[category, 'fit'])
               ax.plot(xs, pdf, self.category_colors[category])
        link_answer_fig, link_answer_ax = plt.subplots(nrows=1, ncols=1, figsize=screensize) #what is the idiomatic way to do this?
        link_answer_ax.set_xlabel("Proportion of engine referers to link referers")
        link_answer_ax.hist(self.link_experimental_vals, normed=True, alpha=exp_alpha, color='blue')
        xs, pdf = normal(self.link_experimental_fit)
        link_answer_ax.plot(xs, pdf, color='blue')
        for category in self.control_categories:
            link_answer_ax.hist(self.link_frame.ix[category,'values'], normed=True, alpha=control_alpha, facecolor=self.category_colors[category])
            xs, pdf = normal(self.link_frame.ix[category, 'fit'])
            link_answer_ax.plot(xs, pdf, color=self.category_colors[category])
        figtitles = {'Engine/Engines Coherence' : engine_coherence_fig, 'Engine/Engines By-Category' : engine_answers_fig, 'Engine/Engines Merged' : engine_answer_fig, 'Enginges/Links By-Category' : link_answers_fig, 'Enginges/Links Merged' : link_answer_fig} #because it is not easy to access figure titles.
        for k,v in figtitles.items(): 
            v.suptitle(k) 
        for k,v in figtitles.items():
            plt.figure(v.number)
            k = k.replace('/','_')
            plt.savefig("results/{}_{}.pdf".format(self.instance['name'], k))
        plt.show()
        engine_coherence_fig.canvas.mpl_disconnect(cid)
    def byTitle(self,title):
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
    def randomSample(self,proportion):
        try:
            # This selects random rows from the database only for titles where the 'n' value for search engine referer rows are all non-zero.
            # It turns out we don't get 0 values, we get simply no entry for other-bing.
            return pd.read_sql(sql='''SELECT DISTINCT title FROM wikiThresh w0
                                      WHERE random() < {} 
                                   '''.format(float(proportion)), con=self.conn)
        except psycopg2.DatabaseError as e:
            print("Failed to read table")
            return None
    def array_of_dicts(self,shape):
        return [ [{} for i in range(shape[0])] for j in range(shape[1])]

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

# what is the sql way to talk about a certain row within an aggregate section? 
#      I think I have it.
def maximal(takeAmount=100):
   " "
   conn = psycopg2.connect(dbname="wiki", user="shalom", host="localhost", password="")
   def maxes(referer):
       return pd.read_sql('''SELECT wOut.title
                             FROM wikiThresh wOut
                             GROUP BY wOut.title
                             ORDER BY (SELECT wIn.n / sum(wOut.n) FROM wikithresh wIn WHERE (wIn.title = wOut.title) AND (wIn.referer = '{0}')) DESC
                             LIMIT {1}
                          '''.format(referer,takeAmount), conn)
   return {referer : maxes(referer) for referer in ('other-google','other-bing','other-yahoo','other-wikipedia')}
