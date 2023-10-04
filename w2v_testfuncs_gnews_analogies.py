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
import plotly
import plotly.figure_factory as ff

from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity as cossim

def cossim2(v1, v2, signed = True):
    """this is the code from the study itself; if it was me, I'd just import
    the cossim function from sklearn - see import call above"""
    c = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    if not signed:
        return abs(c)
    return c

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

def vec_norm(vec):
    return normalize(vec[:,np.newaxis],axis=0).ravel()
    

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
  
#main
# 1. get vectors for words (grouping, individual word, vector) as df
# 2. get cossims (of indiv words, of vector averages)

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

# 4) make a df of the cosine similarities between terms
gund_vecs_cossim = pd.DataFrame(data = cossim(gund_vecs_df), \
                                index = gund_vecs_df.index, \
                                columns = gund_vecs_df.index)

# output cossim matrix to json file
gund_vecs_cossim.to_csv("gund_vecs_cossim_w2v_gnews_CoreTerms.csv")

# #6) get cossim scores for various relationships vis a vis nature/people        
relations = {'nature:people':[('nature', 'people'),
 ('natural', 'people'),
 ('environment', 'people'),
 ('environmental', 'people'),
 ('landscape', 'people'),
 ('wildlife', 'people'),
 ('wilderness', 'people'),
 ('animals', 'people'),
 ('nonhuman', 'people'),
 #('non-human', 'people'),
 ('ecology', 'people'),
 ('ecological', 'people'),
 ('flora', 'people'),
 ('fauna', 'people'),
 ('nature', 'humans'),
 ('natural', 'humans'),
 ('environment', 'humans'),
 ('environmental', 'humans'),
 ('landscape', 'humans'),
 ('wildlife', 'humans'),
 ('wilderness', 'humans'),
 ('animals', 'humans'),
 ('nonhuman', 'humans'),
 #('non-human', 'humans'),
 ('ecology', 'humans'),
 ('ecological', 'humans'),
 ('flora', 'humans'),
 ('fauna', 'humans')],
    'master:servant': [('master','servant')],#('overseer','worker'),
            'parent:child': [('parent','child')],#('father','son'),('mother','daughter'), ('mother','baby'),('father','toddler')],\
            'deity:devotee': [('diety','devotee')],#('god','pious'),('god','acolyte')],\
            'whole:part': [('whole','part')],#('entire','partial')],\
            'ward:warden': [('ward','warden')],#('attendant','attended')],# NB: cared-for is not present in w2v_gnews\
            'supermarket:consumer': [('supermarket','consumer')],#('store','customer')],\
            'provider:recipient': [('provider','recipient')],#('supplier','receiver'),('source','sink')],\
            'benefactor:beneficiary': [('benefactor','beneficiary')],
            'resource:user': [('resource','user')],\
            'playground:child': [('playground','child')],\
            'superior:inferior': [('superior','inferior')],\
            'teacher:student': [('teacher','student')],\
            'hot:cold':[('hot','cold')],\
            'king:queen':[('king','queen')],\
            'pen:pencil':[('pen','pencil')],\
            'gift:receiver':['gift','receiver']
}
    
rel_dist_dict_highlevel = {}

#make a bar graph of analogies' relation to nature:people
comp2 = dir_getter(relations['nature:people'])
nat_people = comp2.copy()
for rel in relations:
    if rel != "nature:people":
        if len(relations[rel]) > 1:
            comp1 = dir_getter(relations[rel])
        else:
            comp1 = word_norm(relations[rel][0][0]) - word_norm(relations[rel][0][1])
        val = cossim2(comp2, comp1)
        print(rel, val)
        if val < 0:
            val = abs(val)
            new_rel = rel.split(":")[1]+":"+rel.split(":")[0]
            rel = new_rel
        rel_dist_dict_highlevel[rel] = val
        
x = pd.Series(data = rel_dist_dict_highlevel)
x.sort_values().plot.barh(title = "Which analogies are like nature:people?")

#make a heatmap of all analogies
rel_dist_dict_high_all = {}

for rel1 in relations['nature:people']:
    rel_dist_dict_high_all[":".join(rel1)] = {}
    comp1 = word_norm(rel1[0]) - word_norm(rel1[1])
    for rel2 in relations:
        if rel2 != "nature:people":
            comp2 = word_norm(relations[rel2][0][0]) - word_norm(relations[rel2][0][1])
            rel_dist_dict_high_all[":".join(rel1)][rel2] = cossim2(comp1, comp2)
new_df2 = pd.DataFrame(rel_dist_dict_high_all).sort_index(ascending = False).sort_index(axis = 1)

new_df2 = new_df2.transpose()

# sorting the df so that all the ref analogies are at the end, and each sub-
# group is sorted by average cosine similarity

new_df2.loc['zAverage score'] = new_df2.mean()

new_df2 = new_df2.sort_index(ascending = False)

new_df2 = new_df2[['resource:user', 'playground:child', 'gift:receiver', 'master:servant',
       'superior:inferior', 'supermarket:consumer', 'teacher:student',
       'provider:recipient', 'parent:child', 'deity:devotee',
       'benefactor:beneficiary', 'whole:part',
       'ward:warden','hot:cold','king:queen', 'pen:pencil']]

x_labels = ['resource:user', 'playground:child', 'gift:receiver', 'master:servant',
       'superior:inferior', 'supermarket:consumer', 'teacher:student',
       'provider:recipient', 'parent:child', 'deity:devotee',
       'benefactor:beneficiary', 'whole:part',
       'ward:warden','(hot:cold)','(king:queen)', '(pen:pencil)']

fig = ff.create_annotated_heatmap(z = new_df2.values, \
                              x = x_labels, \
                              y = list(new_df2.index), \
                              annotation_text = new_df2.values.round(2),\
                              colorscale = ['blue', 'white','red'],\
                              showscale = True)
fig.update_layout(width = 900, height = 800)
plotly.io.write_image(fig, 'nature-people_analogies_w2v-gnews_blue_red.png', format='png')

def analogy_cloud(vec1, topn = 100, excludes = []):
    """takes a vector as input -- ideally one that's an anlogy/relationship between
    two terms/concepts -- and runs through """
    cos_dict = {}
    top_terms = w2v_gnews.index_to_key[:topn]
    if len(excludes) > 0:
        for thing in excludes:
            if thing in top_terms:
                top_terms.remove(thing)
    for i in range(len(top_terms)):
        if i % 100 == 0:
            print(i)
        for j in range(i+1,len(top_terms)):
            rand_vec = w2v_gnews[top_terms[i]] - w2v_gnews[top_terms[j]]
#            print(top_terms[i],top_terms[j])
            cos_dict[(top_terms[i], top_terms[j])] = cossim2(vec1, rand_vec)
    cos_df = pd.DataFrame(data = cos_dict.items())
    word1 = []
    word2 = []
    for i in range(len(cos_df[0])):
        word1.append(cos_df[0][i][0])
        word2.append(cos_df[0][i][1])
    cos_df['word1'] = word1
    cos_df['word2'] = word2
    cos_df.drop(0, axis = 1)
    return cos_df.sort_values(by=1)

nat_people_top5000_df = analogy_cloud(nat_people, topn = 5000, excludes = ["nature", "natural", "environment", "environmental",\
                        "landscape", "wildlife", "wilderness", "animals",\
                        "nonhuman", "non-human", "ecology", "ecological",\
                        "flora", "fauna", 'people','humans'])
    


nat_people_top5000_df.hist(bins = 40, orientation = 'horizontal', figsize = (4,6))

    
    
    