# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 08:51:50 2021

@author: bdemarest
"""

import numpy as np, pandas as pd
import gensim.downloader as api
import plotly
import plotly.figure_factory as ff

from sklearn.metrics.pairwise import cosine_similarity as cossim

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

##main


gund_feats = {'nature': ["nature", "natural", "environment", "environmental",\
                        "landscape", "wildlife", "wilderness", "animals",\
                        "nonhuman", "non-human", "ecology", "ecological",\
                        "flora", "fauna"] ,\
            'wellbeing': ["wellbeing", "well-being", "wellness", "health",\
                         "happiness", "vitality", "healthiness", "health-related",\
                         "healthful", "healthy", "healthier", "contentment",\
                         "quality-of-life", "prosperity"], \
            'spirituality': ["spirituality", "spiritual", "religion", "faith",\
                             "religious", "religiosity", "holiness", "sacred",\
                            "spiritually", "divinity", "divine", "sacredness",\
                            "salvation", "prayer"], \
            'social relations': ["family", "friends", "friend", "neighbours",\
                                 "neighbors", "community", "friendship", "kin",\
                                "familial", "neighborhood", "buddies",\
                                "relatives", "companions", "friendships"], \
            'money': ["money", "cash", "dollars", "income", "wealth", "financial",\
                     "monetary", "wealthy", "dollar", "fortune", "funds",\
                     "investments", "profits", "savings"],\
            "electromagnetism": ["electromagnetism", "electromagnetic", "magnetism",\
                                 "magnetic_fields", "magnetic", "electrons",\
                                "electron", "photons", "ions", "protons",\
                                "electron_spin", "magnet", "electromagnet",\
                                "superconductor"],
            "needs": ["shelter", "food", "necessities", "basic_necessities",\
                    "bare_necessities", "basics", "food_stuffs", "requirements",\
                    "nourishment", "sustenance", "subsistence", "foodstuffs",\
                    "basic_needs", "bare_essentials"],
            "purpose": ["purpose", "goals", "conviction", "meaning", "motivation",\
                        "motivations", "aim", "aims", "mission", "purposes",\
                        "intention", "raison_d'_etre", "rationale", "aspirations"],
            # "motor vehicles": ['auto','automobile','car','vehicle','truck', 'motor'],
            "videogames": ["videogame", "Wii", "Nintendo", "XBox", "Grand_Theft_Auto",\
                          "gaming", "PS3", "Playstation", "Mortal_Kombat",\
                          "videogames", "videogaming", "gamers", "videogamers",\
                          "videogame_consoles"],\
            "pink triangles": ['pink_triangles', 'yarmulkes','tzitzit','kippas',\
                               'kippahs','rainbow_sashes','fezzes','Guy_Fawkes_masks'\
                               'kipa','kaffiyehs','knitted_skullcaps','kipah',\
                               'wear_yarmulkes','kippah'],\
            'outpost': ['outposts','enclave','garrison','beachhead','Nevatim_air',\
                        'frontier','watchtower','hamlet','Baghran_Valley',\
                        'Mitzpe_Yitzhar','Uzbin_Valley','Priestess_Maggie_Q',\
                        'encampment','outpost ']}

# 1) download the embeddings vectors
w2v_gnews = api.load('word2vec-google-news-300')

# 2) get the vecs for the specific terms
gund_vecs = vec_collect(gund_feats, w2v_gnews)

# 3) make a df out of the dict of wordcloud/highlevel and specific terms and vecs
gund_vecs_df = dict_to_df(gund_vecs)

# 3a) make a df out of the mean vecs for each highlevel
gund_vecs_ave_df = pd.DataFrame()
for thing in gund_vecs:
    gund_vecs_ave_df[thing] = mean_vec(list(gund_vecs[thing].values()))

gund_vecs_ave_df = gund_vecs_ave_df.transpose()

# 4) make a df of the cosine similarities between terms
gund_vecs_ave_cossim = pd.DataFrame(data = cossim(gund_vecs_ave_df), \
                                index = gund_vecs_ave_df.index, \
                                columns = gund_vecs_ave_df.index)
    
gund_vecs_ave_cossim = gund_vecs_ave_cossim.sort_values(by = 'wellbeing', ascending=False).sort_values(axis = 1, by = 'wellbeing', ascending = False)

# output cossim matrix to csv file
gund_vecs_ave_cossim.to_csv("gund_vecs_ave_cossim_w2v_gnews.csv")
gund_vecs_ave_cossim = gund_vecs_ave_cossim.iloc[::-1]

#generate heatmap
fig = ff.create_annotated_heatmap(z = gund_vecs_ave_cossim.values, \
                              x = list(gund_vecs_ave_cossim.columns), \
                              y = list(gund_vecs_ave_cossim.index), \
                              annotation_text = gund_vecs_ave_cossim.values.round(2),\
                              colorscale = 'greys',\
                              showscale = True)
fig.update_layout(width = 900, height = 800)
plotly.io.write_image(fig, 'heatmap_nature_concepts.png', format='png')
