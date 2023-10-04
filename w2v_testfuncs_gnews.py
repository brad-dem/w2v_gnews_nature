# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 08:51:50 2021

@author: bdemarest

So what does this code need to do?

1) calculate cosine similarity between two vectors (easy);
2) calculate an average vector from a set of vectors;
"""

import numpy as np, pandas as pd
import gensim.downloader as api

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity as cossim

import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

def cossim2(v1, v2, signed = True):
    """this code calculates cosine similarity between two vectors"""
    c = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    if not signed:
        return abs(c)
    return c

def mean_vec(vec_list):
    return np.add.reduce(vec_list)/len(vec_list)

def vec_collect(terms, embeddings):
    """this function takes a dictionary of lists (high-level terms: [specific 
    instances]) and a dictionary of terms : embedding_vectors, and returns dict
    of struct {high-level : {term : embed array}}"""
    outdict = {}
    for high in terms:
        outdict[high] = {}
        for term in terms[high]:
            try:
                outdict[high][term] = embeddings[term]
            except:
                pass
    return outdict

def dict_to_df(dictry):
    """takes a dict of struct {high-level : {term : array}}, outputs df with 
    2-level index (high-level, term), 1 column per array value"""
    df = pd.DataFrame()
    for high in dictry:
        df2 = df.from_dict(dictry[high], orient='index')
        df2['highlevel'] = [high]*len(df2)
        df = pd.concat([df, df2])
    df.index = [df['highlevel'], df.index]
    del df['highlevel']
    return df
    
def word_norm(vec):
    vec = w2v_gnews[vec]
    return normalize(vec[:,np.newaxis],axis=0).ravel()
    
def dir_getter(list_of_pairs):
    diffs_list = []
    for pair in list_of_pairs:
        try:
            w2v_gnews[pair[0]]
            w2v_gnews[pair[1]]
        except:
            continue
        else:
            norm_0 = normalize(w2v_gnews[pair[0]][:,np.newaxis],axis=0).ravel()
            norm_1 = normalize(w2v_gnews[pair[1]][:,np.newaxis],axis=0).ravel()
            norm_diff = norm_0 - norm_1
            diffs_list.append(norm_diff)
    diffs_list = np.array(diffs_list)
    pca = PCA(n_components = 0.99, random_state = 2020)
    pca.fit(diffs_list)
#    print(pca.explained_variance_ratio_*100)
    return pca.components_[0]

def plot_1d_maker(term_dist_dict, ref_dict):
    # set up the figure
    fig = plt.figure(figsize = [15,3])
    ax = fig.add_subplot(111)
    xmax = 45
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,10)
    fontsize = 18

    # draw lines
    xmin = 0
   
    y = 5
    height = 1

    plt.hlines(y, xmin, xmax)

    term_dist_dict = {k: round(v, 3) for k, v in sorted(term_dist_dict.items(), key=lambda item: item[1])}

    #scaling factor business
    datamin = -0.16
    datamax = 0.19
    
    scale_factor = xmax/abs(datamin-datamax)
    
    plt.vlines(abs(datamin * scale_factor), y - height, y + height)
    plt.annotate(0.0, (abs(datamin * scale_factor),y), xytext = (abs(datamin * scale_factor), y + 1.2), 
            horizontalalignment='center', rotation = 0, color = 'red', fontsize = fontsize) 
    
    keylist = list(term_dist_dict.keys())
    
    # draw a point on the line

    for term in term_dist_dict:
        px = term_dist_dict[term]*scale_factor + abs(datamin * scale_factor)
        if "MEAN" in term:
            plt.plot(px, y, 'ko', ms = 20, mfc = 'g')
        else:
            plt.plot(px, y, 'ko', ms = 10, mfc = 'g')
        plt.annotate(term, (px,y), xytext = (px, y + 1), horizontalalignment='left', rotation = 45, color = 'red')

    # add an arrow
        if keylist.index(term) % 2 == 0:
              y_offset = 4
              if keylist.index(term) % 4 == 0:
                  y_offset = 2
        elif (keylist.index(term) + 1) % 4 == 0:
                  y_offset = -6
        else:
            y_offset = -4
        rot = 0

    # for term in ref_dict:
        try:
            px = ref_dict[term]*5 + 5
        except:
            continue
        else:
            plt.plot(px, y, 'r|', ms = 10, mfc = 'r') 

    # add an arrow
            # plt.annotate(term, (px,y), xytext = (px, y + 1), 
            #           horizontalalignment='left', rotation = 45, color = 'red')

    # add numbers
    plt.text(xmin - 0.1, y, "-0.15\nAsian", horizontalalignment='right', color='red', fontsize = fontsize)
    plt.text(xmax + 0.1, y, "0.15\nwhite", horizontalalignment='left', color='red', fontsize = fontsize)

    plt.axis('off')
    plt.show()

    sns.set(style = 'white')

    fig, axs = plt.subplots(figsize=(10, 1))

    sns.rugplot(ax = axs, x = ref_dict, height = 0.25, lw = 0.25)

    # Remove y axis (by using empty array)
    axs.set_yticks([])

    sns.despine(ax=axs, left=True, offset=2, trim=True)
    
def polar_words(top50000_dist_dict, n = 100):
    for term in top50000_dist_dict:
        terms = term.split("_")
        df = pd.DataFrame(top50000_dist_dict[term].values(), index = top50000_dist_dict[term].keys()).sort_values(by = 0)
        df[:100].to_csv("top100_similar_"+terms[1]+".csv")
        df[-100:].sort_values(by=0, ascending = False).to_csv("top100_similar_"+terms[0]+".csv")
##main
# 1. get vectors for words (grouping, individual word, vector) as df
# 2. get cossims (of indiv words, of vector averages)
    
gund_feats = {'nature': ['nature','natural','environment','environmental',"landscape", "wildlife", "wilderness", "animals", "nonhuman", "non-human", "ecology", "ecological", "flora", "fauna"], \
            'wellbeing': ['wellbeing', 'well-being', 'wellness', 'health', 'happiness', \
                   'vitality', 'healthiness', 'health-related', 'healthful', 'healthy', \
                   'healthier', 'welfare', 'contentment', 'quality-of-life', 'prosperity', \
                   'fulfillment', 'upliftment'], \
            'spirituality': ['spirituality', 'spiritual', 'religion', 'faith', \
                     'religious', 'religiosity', 'holiness', 'sacred', \
                     'spiritually', 'divinity', 'divine', 'soul', 'sacredness', \
                     'salvation','prayer'], \
            'social relations': ['family', 'friends', 'friend','neighbor', \
                                 'neighbors', 'community', 'friendship', \
                                 'kin', 'familial', 'friend', 'friendships', \
                                 'neighbours', 'neighbour'], \
            'money': ['money', 'cash', 'dollars', 'income', 'wealth', 'financial', \
                      'monetary', 'wealthy', 'dollar'],\
            "electromagnetism": ["electromagnetism", "electromagnetic", "magnetism", \
                                 "magnetic_fields", "magnetic", "electricity", \
                                 "electric", "electrons", "electron", "photon", \
                                 "photons", "ions", "particles", "protons", \
                                 "particles", 'electron_spin', "atoms", "atom", \
                                 "antiparticle", "magnet", "electrical", "electromagnet", \
                                 "superconductor"],
            "basic resources": ['shelter', 'food','necessities'],
            "purpose": ['purpose','goals','conviction','meaning'],
            "videogames": ['videogame','Wii','Nintendo','XBox','console','gaming','PS3','Playstation'],\
            "wheels": ['wheels','rolling','axle','steering wheel']\
            }


# 1) download the embeddings vectors
w2v_gnews = api.load('word2vec-google-news-300')

# 2) get the vecs for the specific terms
gund_vecs = vec_collect(gund_feats, w2v_gnews)

# 3) make a df out of the dict of wordcloud/highlevel and specific terms and vecs
gund_vecs_df = dict_to_df(gund_vecs)

# 4) make a df of the cosine similarities between terms
gund_vecs_cossim = pd.DataFrame(data = cossim(gund_vecs_df), \
                                index = gund_vecs_df.index, \
                                columns = gund_vecs_df.index)

# output cossim matrix to json file
gund_vecs_cossim.to_csv("gund_vecs_cossim_w2v_gnews_CoreTerms.csv")


#5) get cossim for nature terms visa vis gender, race, and socioeconomic vectors

grs_nature_dist_dict = {}

top50000_terms = w2v_gnews.index_to_key[:50000]
top50000_dist_dict = {}

grs_dict = {}

grs_dict['whiteVb_black'] = [['whites','blacks'], ['caucasian','negro'],['european','african'],['europeans','africans'],['Caucasians','Blacks'],['Caucasian','Negro'],['European','African'],['Europeans','Africans']]
grs_dict['whiteVa_Asian'] = [['whites','asians'],['caucasian','asian'],['european','japanese'],['europeans','chinese'],['Whites','Asians'],['Caucasian','Asian'],['European','Japanese'],['Europeans','Chinese']]
grs_dict['whiteVh_Hispanic'] = [['whites','latinos'],['caucasian','latinas'],['european','hispanic'],['europeans','hispanics'],['Caucasians','Latinos'],['Caucasian','Latinas'],['European','Hispanic'],['Europeans','Hispanics']]
grs_dict['female_male'] = [['women','men'],['woman','man'],['female','male'],['females','males']]
grs_dict['rich_poor'] = [['rich','poor'],['wealthy','destitute'],['aristocracy','needy'],['nobility','beggars']]

for demog in grs_dict:
    grs_nature_dist_dict[demog] = {}
    top50000_dist_dict[demog] = {}
    if len(grs_dict[demog]) > 1:
        vecs = [word_norm(thing[0])-word_norm(thing[1]) for thing in grs_dict[demog]]
        comp2 = mean_vec(vecs)
    else:
        comp2 = word_norm(grs_dict[demog][0][0])-word_norm(grs_dict[demog][0][1])
    for term in gund_feats['nature']:
        try:
            grs_nature_dist_dict[demog][term] = cossim2(comp2, word_norm(term))
        except KeyError:
            print(term)
    comp1 = dir_getter(gund_feats['nature'])
    grs_nature_dist_dict[demog]['NATURE_MEAN'] = np.mean(list(grs_nature_dist_dict[demog].values()))
    for term in top50000_terms:
        top50000_dist_dict[demog][term] = cossim2(comp2, word_norm(term))
        
polar_words(top50000_dist_dict)

# calculate intervals for 95% confidence   
for demog in grs_nature_dist_dict:
    df = pd.DataFrame(data = grs_nature_dist_dict[demog].values(), index = grs_nature_dist_dict[demog].keys())
    nat_mean = df.loc['NATURE_MEAN'][0].copy()
    df = df.drop('NATURE_MEAN')
    dof = len(df) - 1
    std = np.std(df[0])
    confs = st.t.interval(confidence=0.95, df=dof, loc=nat_mean, scale=st.sem(df[0]))
    print(demog, round(nat_mean, 3), confs)

plot_1d_maker(grs_nature_dist_dict['whiteVb_black'], top50000_dist_dict['whiteVb_black'])
plot_1d_maker(grs_nature_dist_dict['whiteVa_Asian'], top50000_dist_dict['whiteVa_Asian'])
plot_1d_maker(grs_nature_dist_dict['whiteVh_Hispanic'], top50000_dist_dict['whiteVh_Hispanic'])
plot_1d_maker(grs_nature_dist_dict['female_male'], top50000_dist_dict['female_male'])
plot_1d_maker(grs_nature_dist_dict['rich_poor'], top50000_dist_dict['rich_poor'])

    
    
    
    
    
    
    
    
    