"""
=======================================================================================
Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
=======================================================================================

This is an example of applying :class:`sklearn.decomposition.NMF`
and :class:`sklearn.decomposition.LatentDirichletAllocation` on a corpus of documents and
extract additive models of the topic structure of the corpus.
The output is a list of topics, each represented as a list of terms
(weights are not shown).

Non-negative Matrix Factorization is applied with two different objective
functions: the Frobenius norm, and the generalized Kullback-Leibler divergence.
The latter is equivalent to Probabilistic Latent Semantic Indexing.

The default parameters (n_samples / n_features / n_topics) should make
the example runnable in a couple of tens of seconds. You can try to
increase the dimensions of the problem, but be aware that the time
complexity is polynomial in NMF. In LDA, the time complexity is
proportional to (n_samples * iterations).
"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
import csv

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

n_samples = 29651
n_features = 1000
n_topics = 1
n_top_words = 200


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")

with open('train_14.csv', 'rt', encoding='utf-8', errors='ignore') as csvfile:
    reader = csv.reader(csvfile)
    column = [row for row in reader]
label_train = [i[0] for i in column]  # 第一列为class
content_train = [i[1] for i in column]  # 第二列content
# dataset = content_train[11980:13970]
# dataset = content_train[13970:15960]
# dataset = content_train[15960:17950]
# dataset = content_train[17950:19940]
# dataset = content_train[19940:21930]
# dataset = content_train[21930:23920]
# dataset = content_train[23920:25910]
dataset = content_train[25910:26860]
# dataset = content_train[26860:27716]
# dataset = content_train[27716:28685]
# dataset = content_train[28685:29651]
# dataset=content_train[0:8001]

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features)
tfidf = tfidf_vectorizer.fit_transform(dataset)


# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features)

tf = tf_vectorizer.fit_transform(dataset)
# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# Fit the NMF model
print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
      "tf-idf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          solver='cd', max_iter=1000, alpha=.1, l1_ratio=.5).fit(tfidf)


print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# print("Fitting LDA models with tf features, "
#       "n_samples=%d and n_features=%d..."
#       % (n_samples, n_features))
# lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
#                                 learning_method='online',
#                                 learning_offset=50.,
#                                 random_state=0)
#
# lda.fit(tf)
#
#
# # print("\nTopics in LDA model:")
# tf_feature_names = tf_vectorizer.get_feature_names()
# print_top_words(lda, tf_feature_names, n_top_words)
