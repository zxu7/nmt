from utils.utils import load_sgm, Preprocess, WordVector
from model.models import simpleNMT
from keras.preprocessing.sequence import pad_sequences

# load data
data_dir = '/Users/harryxu/school/data/ai_challenger_translation_validation_20170912/translation_validation_20170912/'

en_path = data_dir + 'valid.en-zh.en.sgm'
cn_path = data_dir + 'valid.en-zh.zh.sgm'
en_data = load_sgm(en_path)
cn_data = load_sgm(cn_path)

# tokenization
preprocessor = Preprocess()
preprocessor.fit([en_data, cn_data])
a1, a2 = preprocessor.get_ids()
a1 = pad_sequences(a1, preprocessor.en_maxlen, padding="post", truncating="post")
a2 = pad_sequences(a2, preprocessor.cn_maxlen, padding="post", truncating="post")

# load word vectors
en_wv_path = '/Users/harryxu/NLP/glove/glove.6B.100d.txt'
cn_wv_path = '/Users/harryxu/NLP/glove/kdxf.txt'
WV = WordVector(preprocessor.en_dict, preprocessor.cn_dict)
en_mat, cn_mat = WV.to_matrix(en_wv_path, cn_wv_path)
print(cn_mat.shape)

# fit model
model = simpleNMT(pad_length=preprocessor.cn_maxlen,
                  n_voc_in=cn_mat.shape[0],
                  d_voc_in=cn_mat.shape[1],
                  n_labels=en_mat.shape[1],
                  embedding_learnable=True,
                  encoder_units=2,
                  decoder_units=2,
                  trainable=True,
                  return_probabilities=False,
                  weights=None)

model.fit(a2, a1)
