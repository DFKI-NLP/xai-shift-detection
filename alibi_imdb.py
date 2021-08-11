import numpy as np
from alibi_detect.cd import MMDDrift
from imdb_models_pytorch import train, gradxinput
import os
from bert import BertGradient
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
import re
from data_utils import clean_str

dimensionality_reduction = None
bert = BertGradient()

sample_size = [10, 20, 50, 100, 200, 500]
runs = 15
gxi = False

# load data
imdb = load_dataset('imdb', cache_dir='./data', split='train')\
    .map(lambda x: {'text': clean_str(x['text']).split(' ')})

# load adversarial data
adv_imdb = load_dataset('text', data_files={'train': 'imdb_bert.txt'}, split='train') \
    .filter(lambda x: x['text'].startswith('adv sent')) \
    .map(lambda x: {'text': x['text'].split('):\t')[1].split(' ')})

repr_type = 'embed'
repr_func = {'embed': lambda x: bert.embedding(x),
             'gxi': lambda x: bert.grad_x_input(x),
             'bbsd': lambda x: bert.bbsds(x)}

p_vals = np.zeros((len(sample_size),runs))

for r in range(runs):

    imdb.shuffle()
    imdb_train_test = imdb.train_test_split(test_size=0.1)

    train = imdb_train_test['train']['text']
    val = imdb_train_test['test']['text']

    X_val_embed = repr_func[repr_type](val)

    dr = None
    if dimensionality_reduction == 'srp':
        dr = SparseRandomProjection(n_components=32)
    elif dimensionality_reduction == 'pca':
        dr = PCA(n_components=32)
    if dimensionality_reduction is not None:
        dr.fit(X_tr_embed)
        X_val_embed = pca.transform(X_val_embed)

    for s_i, s in enumerate(sample_size):

        print("{} samples".format(s))

        adv_imdb.shuffle()
        test = adv_imdb[:s]['text']

        X_te_embed = repr_func[repr_type](test)

        if dimensionality_reduction is not None:
            X_te_embed = dr.transform(X_te_embed)

        cd = MMDDrift(X_val_embed.reshape(X_val_embed.shape[0],-1), backend='pytorch', p_val=0.05)
        preds = cd.predict(X_te_embed.reshape(X_te_embed.shape[0],-1))
        print('p-val={}'.format(preds['data']['p_val']))

        p_vals[s_i,r] = preds['data']['p_val']

print('Mean p-vals: {}'.format(np.mean(p_vals,axis=-1)))



