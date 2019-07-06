#coding=utf-8
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import sklearn_crfsuite,joblib
from sklearn_crfsuite import metrics
import parse_data
from collections import defaultdict
class CRF_NER:

    def __init__(self):
        """初始化参数"""
        self.algorithm = "lbfgs"
        self.c1 ="0.1"
        self.c2 = "0.1"
        self.max_iterations = 100
        self.model_path = "model.pkl"
        self.corpus = parse_data.CorpusProcess()  #Corpus 实例
        #self.corpus.pre_process()  
        #self.corpus.initialize() 
        self.model = None

    def initialize_model(self):
        """初始化"""
        algorithm = self.algorithm
        c1 = float(self.c1)
        c2 = float(self.c2)
        max_iterations = int(self.max_iterations)
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2,
                                          max_iterations=max_iterations, all_possible_transitions=True)

    def train(self):
        """训练"""
        self.initialize_model()
        x, y = self.corpus.generator()
        x_train, y_train = x[500:], y[500:]
        x_test, y_test = x[:500], y[:500]
        self.model.fit(x_train, y_train)
        labels = list(self.model.classes_)
        labels.remove('O')
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3))
        self.save_model()

    def predict(self, sentence):
        """预测"""
        self.load_model()
        u_sent = self.corpus.q_to_b(sentence)
        word_lists = [[u'<BOS>']+[c for c in u_sent]+[u'<EOS>']]
        word_grams = [self.corpus.segment_by_window(word_list) for word_list in word_lists]
        features = self.corpus.extract_feature(word_grams)
        y_predict = self.model.predict(features)
        entity = u''
        result=defaultdict(list)
        for index in range(len(y_predict[0])):
            if y_predict[0][index] != u'O':
                if index > 0 and y_predict[0][index][-1] != y_predict[0][index-1][-1]:
                    #entity += u' '
                    if entity:
                        if entity not in result[y_predict[0][index-1]]:
                            result[y_predict[0][index-1]].append(entity)
                        entity=""
                entity += u_sent[index]
            elif entity and entity[-1] != u' ':
                #entity += u' '
                if entity not in result[y_predict[0][index-1]]:
                    result[y_predict[0][index-1]].append(entity)
                entity=""                
        return result

    def load_model(self):
        """加载模型 """
        self.model = joblib.load(self.model_path)

    def save_model(self):
        """保存模型"""
        joblib.dump(self.model, self.model_path)
if __name__=='__main__':
    ner = CRF_NER()
    ner.train()
    print("Train Done")