# -*- coding: utf-8 -*-
import csv
import jieba
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

# 读取训练集
def readtrain():

    with open('train_2.csv', 'rt',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row for row in reader]
        # print(column)
    label_train = [i[0] for i in column] # 第一列为class
    content_train = [i[1] for i in column] # 第二列content
    print ('训练集有 %s 条句子' % len(content_train))
    train = [content_train, label_train]
    # print(train)
    return train
def readtest():
    with open('x_y_test(4).csv', 'rt',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row for row in reader]
        # print(column)
    y_test = [i[0] for i in column]  # 第一列为table
    x_test = [i[1] for i in column]  # 第二列为content
    print('测试集有 %s 条句子' % len(x_test))
    test = [x_test, y_test]
    return test



# 将utf8的列表转换成unicode
def changeListCode(b):
    a = []
    for i in b:
        a.append(i.decode('utf8'))
    return a
# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
        # print(c)
    return c


def estimate(predicted,y_test):
    print('accuracy:', accuracy_score(predicted, y_test))
    print('precision:', precision_score(predicted, y_test, average='macro'))
    # print('precision:',precision_score(predicted, test_label, average=None))
    print('recall:', recall_score(predicted, y_test, average='macro'))
    print('f1:', f1_score(predicted, y_test, average='macro'))
    print('类别标签：', set(predicted))


'''
LG
'''
def clf1():
    clf = LogisticRegression().fit(tfidf, y_train)
    new_tfidf = tfidftransformer.transform(vectorizer.transform(x_test))
    predicted = clf.predict(new_tfidf)
    estimate(predicted,y_test)
    print(predicted)

'''
SGD
'''
def clf2():
    clf = linear_model.SGDClassifier().fit(tfidf, y_train)
    new_tfidf = tfidftransformer.transform(vectorizer.transform(x_test))
    predicted = clf.predict(new_tfidf)
    estimate(predicted,y_test)

'''
DT
'''
def clf3():
    clf = tree.DecisionTreeClassifier().fit(tfidf, y_train)
    new_tfidf = tfidftransformer.transform(vectorizer.transform(x_test))
    predicted = clf.predict(new_tfidf)
    estimate(predicted,y_test)

'''
NB
'''
# 单独预测
# 分类器
def clf4():
    clf = MultinomialNB().fit(tfidf, y_train)
    new_tfidf = tfidftransformer.transform(vectorizer.transform(x_test))
    predicted = clf.predict(new_tfidf)
    estimate(predicted,y_test)

'''
SVC
'''
# 训练和预测一体
#
def clf5():
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=0.99, kernel = 'linear'))])
    text_clf = text_clf.fit(x_train, y_train)
    predicted = text_clf.predict(x_test)
    print(predicted)
    # print(predicted == y_test)
    print ('SVC',np.mean(predicted == y_test))
    estimate(predicted,y_test)
# # print (metrics.confusion_matrix(test_label,predicted)) # 混淆矩阵



# 循环调参
'''
parameters = {'vect__max_df': (0.4, 0.5, 0.6, 0.7),'vect__max_features': (None, 5000, 10000, 15000),
              'tfidf__use_idf': (True, False)}
grid_search = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
grid_search.fit(content, opinion)
best_parameters = dict()
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

'''

if __name__ =="__main__":
    # corpus = ["我 来到 北京 清华大学", "他 来到 了 网易 杭研 大厦", "小明 硕士 毕业 与 中国 科学院"]
    train = readtrain()
    x = train[0]
    y = train[1]
    # test=readtest()
    # x_test=test[0]
    # y_test=test[1]

    # 划分
    # train_content = content[0:771]+content[856:1728]+content[1825:2694]
    # test_content = content[771:856]+content[1728:1825]+content[2694:2791]
    # train_label = label[0:771]+label[856:1728]+label[1825:2694]
    # test_label = label[771:856]+label[1728:1825]+label[2694:2791]

    x_train=x[0:29650]
    y_train=y[0:29650]
    x_test=x[29651:29675]
    y_test=y[29651:29675]
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.0005, random_state=42)

    # 计算权重
    vectorizer = CountVectorizer()
    tfidftransformer = TfidfTransformer()
    tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(x_train))  # 先转换成词频矩阵，再计算TFIDF值
    print('tfidf.shape：', tfidf.shape)

    clf1()

    clf2()
    clf3()
    clf4()
    clf5()

