from train import CRF_NER
import parse_data
import joblib
from collections import defaultdict
class Predict(CRF_NER):
    def __init__(self,):
        self.model_path = "model.pkl"
        self.model = None
        self.corpus = parse_data.CorpusProcess()
    def predict(self, sentence):
        """预测"""
        self.model = joblib.load(self.model_path)
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
if __name__=='__main__':
    ner = Predict()
    maps_ = {u'I_T':"Time",u'I_PER': u'Name', u'I_ORG': u'Group',u'I_LOC': u'Location'}
    result = {}
    while 1:
        sentence = input('ner sentence:').strip()
        ners_result = ner.predict(sentence)
        for k,v in ners_result.items():
            if k in maps_:
                result[maps_.get(k)] = v
        print(result)
        