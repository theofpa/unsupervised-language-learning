from __future__ import absolute_import, division, unicode_literals

import numpy as np
import pickle
import logging
import senteval

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


class dotdict(dict):

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


params_senteval = {'task_path': '',
                   'usepytorch': False,
                   'kfold': 10,
                   'emb_path': '',
                   'tok_path': '',
                   'embeddings': None,
                   'tokenizer': None}

params_senteval = dotdict(params_senteval)


def prepare(params, samples):
    params.embeddings = np.load(params.emb_path)
    params.tokenizer = pickle.load(open(params.tok_path, 'rb'))


def batcher(params, batch):

    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    for sent in batch:

        x1 = [params.tokenizer[word] for word in sent]
        z_batch1 = params.embeddings[x1, :]
        sent_vec = np.mean(z_batch1, axis=0)

        if np.isnan(sent_vec.sum()):
            sent_vec = np.nan_to_num(sent_vec)
        embeddings.append(sent_vec)
    embeddings = np.vstack(embeddings)
    return embeddings


if __name__ == "__main__":

    params_senteval.task_path = ''
    params_senteval.tok_path = 'tokenizer.pickle'
    params_senteval.emb_path = 'embeddings.npy'
    params_senteval.kfold = 10
    se = senteval.engine.SE(params_senteval, batcher, prepare)

    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']

    results = se.eval(transfer_tasks)
    print(results)
