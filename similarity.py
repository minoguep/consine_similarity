#######################################################
## Author: minoguep
##
## Script to illustrate difference between bag of words
## and word embeddings for sentence similarity
#######################################################

import pandas as pd
import numpy as np
import string
import itertools
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#################
### FUNCTIONS ###
#################

# function to remove punctuation from text (input is a string)
def clean_text(sentence):
	
	clean_sentence = "".join(l for l in sentence if l not in string.punctuation)
	
	return clean_sentence

# function to calculate the cosine
def cosine_similarity_calc(vector_1,vector_2):
	
	similarity = np.dot(vector_1,vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
	
	return similarity

# function to calculate cosine similarity using bow representation (input is a dataframe)
def bow_similarity(sentences_df):
	
	# first lets clean the text by removing punctuation
	sentences_df['clean_text'] = sentences_df.apply(lambda row: clean_text(row['sentence_text']), axis=1)
	
	# initialise the bag of words tokeniser and apply it to our clean text
	# this will create vector representations for each word
	count_vec = CountVectorizer()
		
	dtm = count_vec.fit_transform(sentences_df['clean_text']).toarray()
	
	# calculate similarity, this will return an NxN matrix withs 1s across the diagonal
	similarity_df = pd.DataFrame(cosine_similarity(dtm)).reset_index()
	
	# from here we are essentially just going to pivot the similairty matrix to return
	# the data in a format like:
	# | sentence_a | sentence_b | similarity |
	
	# unpivot the similarity df
	df_unpiv = pd.melt(similarity_df, id_vars=['index'])
	
	# get unique combinations of sentance_a and sentance_b
	df_unpiv_unique = (df_unpiv.loc[pd.DataFrame(np.sort(df_unpiv[['index', 'variable']],1),index=df_unpiv.index)
						.drop_duplicates(keep='first')
						.index])

	# remove instances where sentence a == sentence b
	df_unpiv_unique = df_unpiv_unique[df_unpiv_unique['index'] != df_unpiv_unique['variable']]
	
	# now finally join on the original df to get the required output
	# join the text
	df_with_text = pd.merge(df_unpiv_unique, sentences_df.reset_index()
							, left_on='index'
							, right_on='index')
	df_with_text = pd.merge(df_with_text, sentences_df.reset_index()
							, left_on='variable'
							, right_on='index')

	df_with_text = df_with_text.loc[:,['sentence_text_y', 'sentence_text_x', 'value']]
	df_with_text.columns = ['sentence_a', 'sentence_b', 'similarity']
	
	return df_with_text

# function to calculate cosine similarity using word vectors (input is a series)
def embeddings_similarity(sentences):
	
	# first we need to get data into | sentence_a | sentence_b | format
	sentence_pairs = list(itertools.combinations(sentences, 2))
	
	sentence_a = [pair[0] for pair in sentence_pairs]
	sentence_b = [pair[1] for pair in sentence_pairs]
	
	sentence_pairs_df = pd.DataFrame({'sentence_a':sentence_a, 'sentence_b':sentence_b})
	
	# get unique combinations of sentance_a and sentance_b
	sentence_pairs_df = sentence_pairs_df.loc[pd.DataFrame(np.sort(sentence_pairs_df[['sentence_a', 'sentence_b']],1)
														   ,index=sentence_pairs_df.index).drop_duplicates(keep='first').index]

	# remove instances where sentence a == sentence b
	sentence_pairs_df = sentence_pairs_df[sentence_pairs_df['sentence_a'] != sentence_pairs_df['sentence_b']]
	
	# load word embeddings (will use these to convert sentence to vectors)
	# Note you will need to run the following command (from cmd) to download embeddings: 
	# 'python -m spacy download en_core_web_lg'
	embeddings = spacy.load('en_core_web_lg')
	
	# now we are ready to calculate the similarity
	
	sentence_pairs_df['similarity'] = sentence_pairs_df.apply(lambda row: cosine_similarity_calc(embeddings(clean_text(row['sentence_a'])).vector,
																					embeddings(clean_text(row['sentence_b'])).vector), axis=1)
	
	return sentence_pairs_df

############
### MAIN ###
############

# load input data
sample_sentences = pd.read_csv('./data/sample_sentences.txt', sep='|')

# calculate similarity using both methods
bag_of_words_output = bow_similarity(sample_sentences)
word_embeddings_output = embeddings_similarity(sample_sentences['sentence_text'])

# combine outputs
combined_output = pd.merge(word_embeddings_output, bag_of_words_output
						   , left_on=['sentence_a', 'sentence_b']
						   , right_on=['sentence_a', 'sentence_b'])

combined_output.columns = ['sentence_a', 'sentence_b', 'embedding_similarity', 'bow_similarity']

# output as pipe delimited text file
combined_output.to_csv('./output/similarity_output.txt', sep='|')

print('Done!')

sentences = ['Hi, how are you?', 'Hey what\'s up?']
sentences_df = pd.DataFrame({'sentences':sentences})