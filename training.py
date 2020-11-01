from gensim.models import word2vec  #加载库
from gensim.models import Word2Vec
sentences=word2vec.Text8Corpus(r'/home/chenzx/token/CWE590.txt') #加载分好词的文件
#模型训练
print("training...")
model_with_loss = word2vec.Word2Vec(sentences, sg=0, size=128,  window=3,  min_count=5, compute_loss=True, negative=3, sample=0.001, hs=1, workers=4)
#模型保存
print("saving...")
model_with_loss.save(r'/home/chenzx/model/CWE590.model')
training_loss = model_with_loss.get_latest_training_loss()
print(training_loss)

#模型加载
# model = word2vec.load(r'C:\Users\44511\Desktop\testcases\model\test.model')
# print(model['i8*'])
# print(model.most_similar("i8*",topn=10))