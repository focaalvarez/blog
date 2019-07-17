# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:46:11 2019

@author: Juan Felipe Alvarez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



pubs=pd.read_excel('open_pubs.xlsx',sheet_name='open_pubs',converters={'name':str})
cities=pd.read_excel('open_pubs.xlsx',sheet_name='population')


#transform to lower case and transform to strings
pubs['name']=pubs['name'].str.lower()

#create 1 object with all words
text = " ".join(pubs['name'])

#count words in names
pubs['word_count'] = pubs['name'].apply(lambda x: len(str(x).split(" ")))
pubs.hist(column=['word_count'],grid=False,bins=10)
print('The average lenght of the Pub names is ',str(pubs['word_count'].mean()),' words.')


stopwords=['the','ltd','club','inn']

#create word cloud
wordcloud = WordCloud(max_words=150, background_color="white",stopwords=stopwords,normalize_plurals=False).generate(text)
plt.figure(dpi=500)
plt.imshow(wordcloud, interpolation="bicubic")
plt.axis("off")
plt.show()
wordcloud.to_file("pub_wordcloud.png")

#Show the 10 most popular unique names
print('10 most popular unique Pub names in Uk')
pubs['name'].value_counts().head(10)

#load lists
animals=pd.read_excel('animals.xlsx',sheetname='animals',header=None)
animals="|".join(animals[0])
colors=pd.read_excel('animals.xlsx',sheetname='colors',header=None)
colors="|".join(colors[0])
royal=pd.read_excel('animals.xlsx',sheetname='royal',header=None)
royal="|".join(royal[0])
sports=pd.read_excel('animals.xlsx',sheetname='sports',header=None)
sports="|".join(sports[0])

#Calculate the % of times there are rows containing the words in the lists
print(str(pubs['name'][pubs['name'].str.contains(animals)].count()/len(pubs)*100),'% of Pubs have a reference to an Animal')
print(str(pubs['name'][pubs['name'].str.contains(colors)].count()/len(pubs)*100),'% of Pubs have a reference to a Color')
print(str(pubs['name'][pubs['name'].str.contains(sports)].count()/len(pubs)*100),'% of Pubs have a reference to a Sport')
print(str(pubs['name'][pubs['name'].str.contains(royal)].count()/len(pubs)*100),'% of Pubs have a reference to royal vocabulary')

#load population
pubs_by_city=pubs_by_city=pd.pivot_table(pubs,index=['city'],values=['name'],aggfunc='count')
pubs_by_city.sort_values(by='name',ascending=False,inplace=True)
pubs_by_city.reset_index(inplace=True)
pubs_by_city=pubs_by_city.merge(cities,how='left',on='city')
pubs_by_city['pubs_per_1000_persons']=pubs_by_city['name']/pubs_by_city['population']*1000
pubs_by_city.sort_values(by='pubs_per_1000_persons',ascending=False,inplace=True)





