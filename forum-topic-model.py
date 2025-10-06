# -*- coding: utf-8 -*-
'''
This code is:
Copyright 2024 [Names removed for review]
Patent Pending
'''
# ## import libraries
#
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, TopicDiversity
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.models.ctm import ZeroShotTM
#
import pandas as pd
import wordcloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from pathlib import Path
#
#
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
#
import os
import pickle
from scipy import stats


"""# Data Preparation"""

STOPWORD_LIST = '/path/to/StopWords.txt'
MODEL_SAVE_PATH = '/path/to/directory/Data/'	#output data directory for each run
MINIMUM_WORD_PER_DOCUMENT = 50

#dataset locations. CSV expected with one thread per line
local_datasets = {
	'DATASETNAME': '/path/to/dataset.csv',
	'STACK' : '/path/to/stackexchange.csv',
	'KAGGLE' : '/path/to/kaggle.csv',
}

#dataset encoding. utf-8 or iso-8859-1, for example
text_codec = { 
	'DATASETNAME': 'encoding',
	'STACK' : 'encoding',
	'KAGGLE' : 'encoding',
}

#this variable is set this way to differentiate between a local run and a possible cloud run
datasets = local_datasets

def load_and_clean_data(data_path:str, codec: str='utf-8')-> list:
  _documents = [line.strip() for line in open(data_path, encoding=codec).readlines()]
  cleaned_documents = []
  for i in range(0, len(_documents)):
    if len(_documents[i].split()) >= 50 and len(_documents[i].split()) <= 10000:
      cleaned_documents.append(_documents[i])
  return cleaned_documents

## Stopword
with open(STOPWORD_LIST, "r") as file_content:
  LoadedStopWordsCombined = file_content.read().split("\n")
  # print(LoadedStopWordsCombined)

## Load and Clean Data
documents = {}
for key, value in datasets.items():
  print(f"loading file {key}...")
  documents[key] = load_and_clean_data(value, codec=text_codec[key])

"""### Sanity Check"""
#check the number of items in the document
len(documents['DATASETNAME'])

# No document should less than 50 words
for k, v in documents.items():
  for i in range(0, len(v)):
    assert len(v[i].split()) >= MINIMUM_WORD_PER_DOCUMENT, "Error: Short Text Included"

"""# Training the model
We will use STACK data to train our model
"""
#Adjust these parameters for the runs
VOCABULARY_SIZE = 5000  # NOTE: use 5000 for local training
NUM_TOPICS = 50        # NOTE: use 50 for local training
NUM_EPOCHS = 100        # NOTE: use 50 for local training
SBERT_MODEL_NAME = "paraphrase-distilroberta-base-v2"   # alternative: "all-mpnet-base-v2"
SAMPLE_SIZE = -1     # NOTE: USE SAMPLE_DATASET for training in google cloud for speedup; otherwise, use full documents by SETTING to -1
# SAMPLE_DATASET = documents['STACK'][:100] 
RESULT_FOLDER = Path.cwd() / MODEL_SAVE_PATH / f"ZSTM_voc{VOCABULARY_SIZE}_topics{NUM_TOPICS}_epochs{NUM_EPOCHS}_{SBERT_MODEL_NAME}"	#where the model is stored
WORDCLOUD_FOLDER = RESULT_FOLDER / "WordCloud"		#location for wordcloud output
PLOT_FOLDER = RESULT_FOLDER / "Plots"			#location for graphic plots

SKEWNESS_THRESHOLD = 1  ## NOTE: if sample size is too small, high threshold might yield empty results; USE 2 for local env
CUTOFF = 0.2		#cutoff criterion is set here

# Create the OUTPUT FOLDERS
RESULT_FOLDER.mkdir(parents = True, exist_ok = True)
print("Creating WordCloud folder ...")
WORDCLOUD_FOLDER.mkdir(parents = True, exist_ok = True)
print("Creating Plot folder ...")
PLOT_FOLDER.mkdir(parents = True, exist_ok = True)

# Remove Stopwords from documents
training = documents['STACK'] if SAMPLE_SIZE == -1 else documents['STACK'][:SAMPLE_SIZE]
sp = WhiteSpacePreprocessingStopwords(documents = training, stopwords_list = LoadedStopWordsCombined, vocabulary_size = VOCABULARY_SIZE, remove_numbers = True)
stack_preprocessed_documents, stack_unpreprocessed_documents, vocab, returned_indices = sp.preprocess()

# Create Embeddings
tp = TopicModelDataPreparation(SBERT_MODEL_NAME)
stack_training_dataset = tp.fit(text_for_contextual=stack_unpreprocessed_documents, text_for_bow=stack_preprocessed_documents)

# Training ZSTM
zstm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=NUM_TOPICS, num_epochs=NUM_EPOCHS, num_data_loader_workers = 0)  ### sometimes different bert input size 512; n_components = number of topics #only for windows: 
zstm.fit(stack_training_dataset)

# Save the model
pickle.dump(zstm, open(os.path.join(RESULT_FOLDER, f"ZSTM_voc{VOCABULARY_SIZE}_topics{NUM_TOPICS}_epochs{NUM_EPOCHS}_{SBERT_MODEL_NAME}.pkl"), 'wb'))

"""### Evaluate Model"""
NUM_WORDS_PER_TOPIC = 25
topics_word_score_list = []
for topic_id in range(NUM_TOPICS):
    word_score_list = zstm.get_word_distribution_by_topic_id(topic_id)[:NUM_WORDS_PER_TOPIC]
    word_score_dict = {tup[0]: tup[1] for tup in word_score_list}   
    topics_word_score_list.append(word_score_dict)

def eval_model(model, documents):
  texts = [a.split() for a in documents]

  ## Topic-Word list: TOP words for each topics
  topics_list = [ list(d.keys()) for d in topics_word_score_list ]

  #Calculate scores
  npmi_score = CoherenceNPMI(texts=texts, topics=model.get_topic_lists(10)).score()
  topic_div_score = TopicDiversity(topics = topics_list).score(topk = NUM_WORDS_PER_TOPIC)
  topic_quality_score = npmi_score * topic_div_score

  result = f"NPMI: {npmi_score}, Topic Diversity Score: {topic_div_score}, Topic Quality Score: {topic_quality_score}"
  print(result)
  with open(os.path.join(RESULT_FOLDER, f"model_coherence_voc{VOCABULARY_SIZE}_topics{NUM_TOPICS}_epochs{NUM_EPOCHS}_{SBERT_MODEL_NAME}.pkl.txt"), 'w') as res:
    res.write(result)

eval_model(zstm, stack_preprocessed_documents)

"""# Prediction

### Load the Model
"""
#this model was the one that was calculated and trained. now we will use it again
MODEL_TO_USE = RESULT_FOLDER / f"ZSTM_voc{VOCABULARY_SIZE}_topics{NUM_TOPICS}_epochs{NUM_EPOCHS}_{SBERT_MODEL_NAME}.pkl"


zstm_loaded = pickle.load(open(os.path.join(MODEL_SAVE_PATH, MODEL_TO_USE), 'rb'))

eval_model(zstm_loaded, stack_preprocessed_documents)

"""### Run Prediction on StackEx """

NUM_SAMPLES = 100   # NOTE: use 1 for cloud training otherwise use 150 for local training
stack_topics_predictions = zstm.get_doc_topic_distribution(stack_training_dataset, NUM_SAMPLES) # the higher n the better are the probabilities
np.savetxt(os.path.join(RESULT_FOLDER, f"topic_probabilities_Stack_voc{VOCABULARY_SIZE}_topics{NUM_TOPICS}_epochs{NUM_EPOCHS}_{SBERT_MODEL_NAME}.csv"), stack_topics_predictions, delimiter = ",")

for topic_id, word_score_dict in enumerate(topics_word_score_list):
  word_score_dict_20 = dict(sorted(word_score_dict.items(), key=lambda x: x[1], reverse=True)[:20])
  plt.ioff()
  plt.figure(figsize=(10, 4), dpi=400)
  plt.axis("off")
  plt.imshow(wordcloud.WordCloud(width=1000, height=400, background_color='black').generate_from_frequencies(word_score_dict_20))
  plt.title("Displaying Topic " + str(topic_id), loc='center', fontsize=24)
  plt.savefig(os.path.join(WORDCLOUD_FOLDER, f"topic_{topic_id}.png"), facecolor = "white")
  plt.close()

def filter_threshold(topics_predictions, threshold):
  """
  Skewness
  """
  skewness_list = []
  for a in topics_predictions:
    skewness_list.append(stats.skew(a))
 
  idx_filt_pred = {}
  for skewness in skewness_list:
    if skewness > threshold:
      index = skewness_list.index(skewness)
      idx_filt_pred[index] = list(topics_predictions[index])
  return idx_filt_pred


# def return_top_topics(idx_filt_pred, unpreprocessed_documents, cutoff):
#   """
#   return top topics
#   """
#   idx_tp_found = list(idx_filt_pred.keys())
#   dictionary = [{ "post number" : key, "top_topics": [ value.index(v) for v in value if v > cutoff], "post content" : unpreprocessed_documents[key]} for key, value in idx_filt_pred.items() ]
#   tmp_dict = [{ "post number" : unpreprocessed_documents.index(v), "top_topics": [], "post content" : v } for v in unpreprocessed_documents if v not in idx_tp_found ]
#   dictionary.extend(tmp_dict)
#   dictionary = sorted(dictionary, key = lambda x:x["post number"])
#   return dictionary

def topic_tune_count(predictions, model, unpreprocessed_documents, skewness_threshold, cutoff, name):
  ## define cutoff criterium based on skeweness 
  idx_filter_pred = filter_threshold(predictions, skewness_threshold) 

  #determine top topics
  # dictionary = return_top_topics(idx_filter_pred, unpreprocessed_documents, cutoff)
  idx_tp_found = list(idx_filter_pred.keys())
  dictionary = [{ "post number" : key, "top_topics": [ value.index(v) for v in value if v > cutoff], "post content" : unpreprocessed_documents[key]} for key, value in idx_filter_pred.items() ]

  dictionary = sorted(dictionary, key = lambda x:x["post number"])

  df_top = pd.DataFrame(dictionary)
  df_top.to_csv(os.path.join(RESULT_FOLDER, f"{name}_top_topics_per_post.csv"), index=False)

  #get post count
  with open(f"{name}_post_count.csv", "w+") as fp:
    fp.write(str(len(unpreprocessed_documents)) + '\n')

  topic_count = dict(Counter([ topic_number for t in dictionary for topic_number in t['top_topics'] ]))
  data = {'topic': list(topic_count.keys()), 'count': list(topic_count.values())}

  df = pd.DataFrame.from_dict(data)
  df_topic_count = pd.DataFrame.from_dict(data).set_index("topic").sort_index()

  ## Allign topic words to dataframe
  df['words'] = df.topic.apply(lambda x: ' ,'.join(map(str, zstm_loaded.get_topic_lists(10)[x]))) ## 40 = number of words to return per topic, default 10.
  df.to_csv(os.path.join(RESULT_FOLDER, f"{name}_topics.csv"), index = False)
  return df_topic_count

df_stack = topic_tune_count(stack_topics_predictions, zstm_loaded, stack_unpreprocessed_documents, SKEWNESS_THRESHOLD, CUTOFF, "STACK")

"""### Prediction on unseen KAGGLE Dataset"""

def make_prediction(model, name, unseen_documents, skewness_threshold, cutoff, n_samples):
  """
    @n_sample: how many times to sample the distribution (see the documentation)
  """
  testing_dataset = tp.transform(unseen_documents)  # tokenize unseen dataset
 
  predictions = model.get_thetas(testing_dataset, n_samples)
  np.savetxt(os.path.join(RESULT_FOLDER, f"topic_probabilities_{name}_voc{VOCABULARY_SIZE}_topics{NUM_TOPICS}_epochs{NUM_EPOCHS}_{SBERT_MODEL_NAME}.csv"), predictions, delimiter = ",")

  count_df = topic_tune_count(predictions, model, unseen_documents, skewness_threshold, cutoff, name)
  return count_df

testing = documents['KAGGLE'] if SAMPLE_SIZE == -1 else documents['KAGGLE'][:SAMPLE_SIZE]
df_kaggle = make_prediction(zstm_loaded, "KAGGLE", testing, SKEWNESS_THRESHOLD, CUTOFF, NUM_SAMPLES)

"""# Evaluation"""

# Enumerate all datasets
#this will evaluate the dataset against the model and return the topics and counts
for name, v in documents.items():
  if name == 'STACK' or name == 'KAGGLE':
    continue
  unseen = v if SAMPLE_SIZE == -1 else v[:SAMPLE_SIZE]
  df_count = make_prediction(zstm_loaded, name, unseen, SKEWNESS_THRESHOLD, CUTOFF, NUM_SAMPLES)
  compare_all = pd.concat([df_stack, df_kaggle, df_count], axis =1) 
  compare_all.columns = [ "STACK", "KAGGLE", name ]
  compare_all.div(compare_all.sum(axis=0), axis=1)
  compare_all.to_csv(os.path.join(RESULT_FOLDER, f"{name}_component_all_tops.csv"), index = False)

  PLOT_UNSEEN_FOLDER = PLOT_FOLDER / name
  PLOT_UNSEEN_FOLDER.mkdir(parents = True, exist_ok = True)

  plt.ioff()
  compare_all.div(compare_all.sum(axis=0), axis=1).plot.bar( )
  fig = plt.gcf()
  plt.ylabel('normalized frequencies',fontsize=18)
  plt.xlabel('topic number', fontsize=18)
  plt.legend(fontsize=18)
  fig.set_size_inches(25, 15)
  fig.savefig(os.path.join(PLOT_UNSEEN_FOLDER, f"compare_all_{name}_voc{VOCABULARY_SIZE}_topics{NUM_TOPICS}_epochs{NUM_EPOCHS}_{SBERT_MODEL_NAME}.png"), facecolor = "white")
  plt.close(fig)

  for topic_id, word_score_dict in enumerate(topics_word_score_list):
    compare_all_div = compare_all.div(compare_all.sum(axis=0), axis=1)
    fig, axes = plt.subplots(1,2, figsize=(15, 5) , dpi = 400)
    plt.ioff()
    sns.barplot(ax = axes[0], data = compare_all_div.iloc[[topic_id]])
    axes[0].set(xlabel='dataset', ylabel='% of threads where\n Topic_{} was the most probable'.format(topic_id))

    word_score_dict_20 = dict(sorted(word_score_dict.items(), key=lambda x: x[1], reverse=True)[:20])
    cloud = wordcloud.WordCloud(width=1500, height=1300,background_color='black').generate_from_frequencies(word_score_dict_20)
    plt.imshow(cloud)
    plt.suptitle('Topic_{}'.format(topic_id))
    axes[1].axis('off')
   
    plt.savefig(os.path.join(PLOT_UNSEEN_FOLDER, f"topic_{name}_{topic_id}.png"), facecolor = "white")
    plt.close(fig)
