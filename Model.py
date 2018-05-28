from whoosh.analysis import StandardAnalyzer
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import logging
import time
import numpy
import zarr
import pickle
from Plotting import Plotter


logger = logging.getLogger('ULL')
logging.basicConfig(level=logging.INFO)


class Corpus:

    def __init__(self, files):

        self._content = self._read(files=files)

    def _read(self, files):
        pass

    @property
    def content(self):
        return self._content


class TrainCorpus(Corpus):

    def __init__(self, files, padding, filter):

        Corpus.__init__(self, files)

        self._preprocessor = StandardAnalyzer()
        self._vocabulary = self._build_vocabulary(padding=padding, filter=filter)
        self._n_context_words = None
        self._window_size = None

    def __str__(self):
        return 'Corpus, {voc_size!s} unique tokens'.format(voc_size=len(self._vocabulary))

    def __repr__(self):
        return 'Class {name!r}'.format(name=__class__.__name__)

    def __getitem__(self, item):
        return self._vocabulary[item]

    def __setitem__(self, key, value):
        self._vocabulary[key] = value

    @property
    def n_context_words(self):
        return self._n_context_words

    @property
    def window_size(self):
        return self._window_size

    @property
    def sentences(self):

        for corpus in self._content:
            corpus = corpus.split('\n')
            for sentence in corpus:
                sentence = list(filter(None, ([word.text for word in self._preprocessor(sentence, removestops=False)])))
                yield sentence

    @property
    def vocabulary(self):
        return self._vocabulary

    def _read(self, files):
        content = []
        for file in files:
            logger.info('Reading File {0}'.format(file))
            with open(file, 'r') as f:
                content.append(f.read())
        return content

    def _build_vocabulary(self, padding, filter):

        logger.info('Building Vocabulary, Sentences')

        words = []

        for raw_content in self._content:
            words.extend([token.text for token in self._preprocessor(raw_content, removestops=False)])

        vocabulary = {}
        vocabulary['UNK'] = 0
        
        if padding:
            vocabulary['PAD'] = 1    

        for word in words:
            if word not in vocabulary:
                if filter:
                    if len(vocabulary) <= filter:
                        vocabulary[word] = len(vocabulary)
                else:
                    vocabulary[word] = len(vocabulary)

        return vocabulary

    @property
    def contexts(self, window_size=2):

        contexts = {}

        n_context_words = 0
        for sentence in self.sentences:
            if len(sentence) > window_size*2 + 1:
                for idx in range(window_size, len(sentence)-window_size):
                    context = sentence[idx - window_size:idx] + sentence[idx + 1:idx + 1 + window_size]
                    if sentence[idx] not in contexts:
                        contexts[sentence[idx]] = []
                    contexts[sentence[idx]].append(context)
                    n_context_words += len(context)

        self._n_context_words = n_context_words
        self._window_size = window_size

        return contexts


class Featurizer:

    def __init__(self, train_corpus):

        self._train_data = train_corpus

    @property
    def train_data(self):
        return self._train_data

    def context_words2features(self, mode, output):

        logger.info('Building Training Data, Labels from Contexts')

        word_contexts = self._train_data.contexts
        n_context_words = self._train_data._n_context_words
        window_size = self._train_data._window_size

        z = zarr.open(output, 'w')
        z.create_dataset('train_data', shape=(n_context_words, ), chunks=(256, ), dtype='i')
        z.create_dataset('labels', shape=(n_context_words, ), chunks=(256,), dtype='i')
        counter = 0
        for word, contexts in word_contexts.items():
            logger.info('Processing word {word}, contexts {size}'.format(word=word, size=len(contexts)))
            train_data = []
            labels = []
            word_idx = self._train_data[word]
            for context in contexts:
                for context_word in context:
                    train_data.append(word_idx)
                    labels.append(self._train_data[context_word])
            z['train_data'][counter:counter+len(contexts)*window_size*2] = numpy.array(train_data, dtype=numpy.int32)
            z['labels'][counter:counter + len(contexts) * window_size * 2] = numpy.array(labels, dtype=numpy.int32)
            counter += len(contexts) * window_size * 2


class Skipgram(nn.Module):

    def __init__(self, n_features, n_layers, n_hidden):

        super(Skipgram, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_features = n_features

        self.embedding0 = nn.Embedding(self.n_features, self.n_hidden)
        self.linear1 = nn.Linear(self.n_hidden, self.n_features)


    def forward(self, data):

        x = self.embedding0(data)
        x = self.linear1(x)

        return x

    def train_network(self, data_dir, epochs, batch_size, weight_decay, lr, output_dir):

        self.cuda()
        self.train(True)

        data = zarr.open(data_dir, 'r')

        train_data = data['train_data']
        labels = data['labels']

        n_batches = round(train_data.shape[0]/batch_size)
        opt = optim.Adam(self.parameters(), weight_decay=weight_decay, lr=lr)
        loss_f = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

        losses = []

        for epoch in range(epochs):

            start = time.time()
            scheduler.step()
            avg_loss = numpy.zeros((1,))
            n_samples = 1

            for idx in range(n_batches):

                opt.zero_grad()
                train_batch = train_data[idx*batch_size:idx*batch_size+batch_size]
                label_batch = labels[idx*batch_size:idx*batch_size+batch_size]
                train_batch = Variable(torch.LongTensor(train_batch)).cuda()
                label_batch = Variable(torch.LongTensor(label_batch), requires_grad=False).cuda()
                output = self(train_batch)
                loss = loss_f(output, label_batch)
                loss.backward()
                opt.step()
                n_samples += 1
                avg_loss += numpy.round(loss.cpu().data.numpy(), 3)

            end = time.time()
            avg_loss /= n_samples
            losses.append(avg_loss)
            logger.info('Epoch {0}, Average Loss {1}, Minutes spent {2}'
                    .format(epoch + 1, round(avg_loss.data[0], 4), round((end-start)/60., 2)))

        Plotter.plot_training(epochs=epochs,
                              losses=losses,
                              embedding_dim=self.n_hidden,
                              output=output_dir)

    @property
    def embeddings(self):
        parameters = list(self.parameters())
        embeddings = parameters[0].cpu().data.numpy()
        return embeddings


def run(model, output_dir, epochs, emb_dimensions, lr, batch_size, weight_decay):

    start = time.time()

    if model == 'skipgram':

        train_corpus = TrainCorpus(files=['hansards/training.en'], padding=True, filter=None)

        featurizer = Featurizer(train_corpus)
        featurizer.context_words2features(mode='normal',
                                          output=output_dir + '/' + 'data.zarr')
        skipgram = Skipgram(n_layers=3,
                            n_hidden=emb_dimensions,
                            n_features=len(train_corpus.vocabulary))
        skipgram.train_network(data_dir=output_dir + '/' + 'data.zarr',
                               epochs=epochs,
                               batch_size=batch_size,
                               weight_decay=weight_decay,
                               lr=lr,
                               output_dir=output_dir)

        numpy.save('embeddings', skipgram.embeddings)
        pickle.dump(train_corpus.vocabulary, open('tokenizer.pickle', 'wb'))

    else:
        raise NotImplementedError

    end = time.time()
    logger.info('Finished Run, Time Elapsed {0} Minutes'.format(round((end - start) / 60, 2)))

    return skipgram

if __name__ == '__main__':

    #skipgram, bayes, embed
    run(model='skipgram',
        output_dir='/home/oem/PycharmProjects/ull3',
        epochs=20,
        emb_dimensions=300,
        lr=0.005,
        batch_size=256,
        weight_decay=0.0001)
