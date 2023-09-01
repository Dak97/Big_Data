from pyspark.mllib.classification import SVMWithSGD, SVMModel, LogisticRegressionWithSGD, LogisticRegressionModel, LogisticRegressionWithLBFGS, NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import IDF, HashingTF
from enum import Enum

PATH_MODELS = 'models/'
PATH_DATASET = 'datasets/'

models = Enum('Models', ['SVM', 'LOGREG','NAIBAY',])

MODELS = {
    models.SVM : 'SVM/',
    models.LOGREG: 'LOGREG/',
    models.NAIBAY: 'NAIBAY/'
}

def get_path(model_name):
    return PATH_MODELS + MODELS[model_name]

def get_model_class(model_name, load=False):
    if model_name == models.SVM:
        return SVMModel if load else SVMWithSGD
    if model_name == models.LOGREG:
        return LogisticRegressionModel if load else LogisticRegressionWithLBFGS
    if model_name == models.NAIBAY:
        return NaiveBayesModel if load else NaiveBayes
    
def extract_feature_from_text(rdd, train=True):
    # split positive and negative
    rdd_positive_samples = rdd.filter(lambda x: x['label']==1)
    rdd_negative_samples = rdd.filter(lambda x: x['label']==0)

    # remove labels
    rdd_positive_text = rdd_positive_samples.map(lambda x: x['text'])
    rdd_negative_text = rdd_negative_samples.map(lambda x: x['text'])

    tf = HashingTF(numFeatures=2000)
    rdd_positive_text_tf = rdd_positive_text.map(lambda x: tf.transform(x))
    rdd_negative_text_tf = rdd_negative_text.map(lambda x: tf.transform(x))

    
    idf = IDF().fit(rdd_positive_text_tf.union(rdd_negative_text_tf))

    rdd_positive_text_features = idf.transform(rdd_positive_text_tf)
    rdd_negative_text_features = idf.transform(rdd_negative_text_tf)

    if train:
        rdd_positive_tr = rdd_positive_text_features.map(lambda x: LabeledPoint(1, x))
        rdd_negative_tr = rdd_negative_text_features.map(lambda x: LabeledPoint(0, x))
    else:
        rdd_positive_tr = rdd_positive_text_features.map(lambda x: (1, x))
        rdd_negative_tr = rdd_negative_text_features.map(lambda x: (0, x))

    rdd_train = rdd_positive_tr.union(rdd_negative_tr)

    return rdd_train