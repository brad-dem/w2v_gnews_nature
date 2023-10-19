# w2v_gnews_nature
Code for reproducing findings of "Nature is resource, playground, and gift: What artificial intelligence reveals about human–nature relationships",  
Rachelle K. Gould, Bradford Demarest, Adrian Ivakhiv, & Nicholas Cheney, a scientific study of relationships between nature, culture, gender,
and wellbeing.

## dependencies
python3
pip install numpy scipy pandas gensim plotly sklearn matplotlib seaborn

## demos
Information about each of the files is supplied here:
1) w2v_testfuncs_gnews.py:
   i) read in Google news corpus word2vec model
   ii) extract vectors for the top 50,000 terms
   iii) create a bi-polar dimension based on each of the five concept pairs:
       black-white;
       male-female;
       Asian-white;
       Latino-white;
       rich-poor;
   iv) calculate cosine similarity for top 50,000 terms from the word2vec model, re: each of the dimensions
   v) output rug plot and rudimentary scale for visualization

2) w2v_testfuncs_gnews_analogies.py
   i) read in Google news corpus word2vec model
   ii) extract vectors and create bi-polar dimensions for analogy pairs
   iii) calculate cosine similarity of each bi-polar dimensional pair
   iv) create bar chart showing cosine simliarity between each analogy pair and nature:people
   v) create histogram of cosines between nature: people and random word-pairs taken from the top 5000 terms
   
3) w2v_testfuncs_vecaves.py
   i) read in Google news corpus word2vec model
   ii) extract vectors for sets of words per concept
   iii) calculate mean concept vectors from each word set
   iv) calculate cosine similarities between concept vectors
   v) output heatmap
