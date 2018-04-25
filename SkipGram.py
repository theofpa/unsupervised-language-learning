from whoosh.analysis import StandardAnalyzer
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
import matplotlib.pyplot as plt
import logging
import time
import numpy

logger = logging.getLogger('ULL')
logging.basicConfig(level=logging.INFO)


class Corpus:

    def __init__(self, file):

        self._content = self._read(file=file)
        self._preprocessor = StandardAnalyzer()
        self._sentences = self._content2sentences()
        self._vocabulary = self._get_vocabulary()
        self._n_contexts = None
        self._window_size = None

    def __repr__(self):
        return 'Corpus, ' + str(len(self._vocabulary)) + ' Tokens, ' + str(len(self._sentences)) + ' Sentences. '

    def __getitem__(self, item):
        return self._vocabulary[item]

    @property
    def n_contexts(self):
        return self._n_contexts

    @property
    def window_size(self):
        return self._window_size

    @property
    def sentences(self):
        return self._sentences

    @property
    def vocabulary(self):
        return self._vocabulary

    def _read(self, file):
        logger.info('Reading File {0}'.format(file))
        with open(file, 'r') as f:
            return f.read()

    def _content2sentences(self):
        logger.info('Building Sentences')
        sentences = self._content.split('\n')
        processed_sentences = list(filter(None, ([word.text for word in self._preprocessor(sentence)]
                                                 for sentence in sentences)))
        return processed_sentences

    def _get_vocabulary(self):
        logger.info('Building Vocabulary')
        words = [word.text for word in self._preprocessor(self._content)]
        vocabulary = {}
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)
        return vocabulary

    def get_contexts(self, window_size=2):
        logger.info('Building Contexts, Window Size {0}'.format(window_size))
        contexts = {}
        n_contexts = 0
        for sentence in self._sentences:
            if len(sentence) > window_size*2 + 1:
                for idx in range(window_size, len(sentence)-window_size):
                    context = sentence[idx - window_size:idx] + sentence[idx + 1:idx + 1 + window_size]
                    if sentence[idx] not in contexts:
                        contexts[sentence[idx]] = context
                    elif sentence[idx]:
                        contexts[sentence[idx]].extend(context)
                    n_contexts += len(context)

        self._n_contexts = n_contexts
        self._window_size = window_size

        return contexts


class Featurizer:

    def __init__(self, corpus):
        self._data = corpus

    @property
    def data(self):
        return self._data

    def vocabulary2one_hot(self):
        logger.info('Building OneHot Vectors')
        id = list(self._data.vocabulary.values())
        size = len(id)
        tensor = torch.FloatTensor([[0 for _ in range(0, size)] for _ in range(0, size)])
        tensor[id, id] += 1
        return tensor

    def contexts2features(self):

        one_hot = self.vocabulary2one_hot()
        contexts = self._data.get_contexts()
        n_contexts = self._data._n_contexts
        n_features = len(self._data.vocabulary)

        logger.info('Building Training Data, Labels from Contexts')

        train_data = torch.FloatTensor([[0 for _ in range(n_features)] for _ in range(n_contexts)])
        labels = torch.LongTensor([0 for _ in range(n_contexts)])
        counter = 0
        for word, context_words in contexts.items():
            word_vector = one_hot[self._data[word], :]
            for context_word in context_words:
                train_data[counter, :] = word_vector
                labels[counter] = self._data[context_word]
                counter += 1

        return train_data, labels


class Plotter:

    @staticmethod
    def plot_training(epochs, losses, n_hidden):
        plt.figure()
        plt.title('WordEmbeddings')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot([i for i in range(epochs)], losses, 'r', label='WordEmbeddings %d' % n_hidden)
        plt.legend()
        plt.grid(True)
        plt.show()


class Network(nn.Module):

    def __init__(self, n_features, n_layers, n_hidden):

        super(Network, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_features = n_features

        self.linear0 = nn.Linear(self.n_features, self.n_hidden)
        self.linear1 = nn.Linear(self.n_hidden, self.n_features)

    def forward(self, data):

        x = self.linear0(data)
        x = self.linear1(x)

        return x

    def train_network(self, train_data, labels, epochs, batch_size, weight_decay, lr):
        self.cuda()
        self.train(True)
        n_batches = round(train_data.shape[0]/batch_size)
        opt = optim.Adam(self.parameters(), weight_decay=weight_decay, lr=lr)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
        losses = []
        for epoch in range(epochs):
            scheduler.step()
            avg_loss = numpy.zeros((1,))
            n_samples = 1

            for idx in range(0, n_batches):
                opt.zero_grad()
                train_batch = train_data[idx*batch_size:idx*batch_size+batch_size, :]
                label_batch = labels[idx*batch_size:idx*batch_size+batch_size]
                train_batch = Variable(train_batch, requires_grad=True).cuda()
                label_batch = Variable(label_batch, requires_grad=False).cuda()
                output = self(train_batch)
                loss = nn.CrossEntropyLoss()(output, label_batch)
                loss.backward()
                opt.step()
                n_samples += 1
                avg_loss += numpy.round(loss.cpu().data.numpy(), 3)

            avg_loss /= n_samples
            losses.append(avg_loss)
            logger.info('Epoch {0}, Average Loss {1}'
                    .format(epoch + 1, round(avg_loss.data[0], 4)))

        Plotter.plot_training(epochs=epochs,
                              losses=losses,
                              n_hidden=self.n_hidden)

    def evaluate(self, corpus, featurizer):
        self.train(False)
        #one_hots = featurizer.vocabulary2one_hot()
        #contexts = featurizer._data.get_contexts()['mr']
        #word = corpus['mr']
        #vector = one_hots[word, :]
        #vector = Variable(vector).cuda()
        #output = self(vector)
        #output = output.cpu().data.numpy().flatten()


if __name__ == '__main__':
    start = time.time()
    corpus = Corpus(file='wa/test.en')
    featurizer = Featurizer(corpus)
    train_data, labels = featurizer.contexts2features()

    network = Network(n_layers=3,
                      n_hidden=500,
                      n_features=train_data.shape[1])

    network.train_network(train_data=train_data,
                          labels=labels,
                          epochs=40,
                          batch_size=256,
                          weight_decay=0.0001,
                          lr=0.005)

    network.evaluate(corpus, featurizer)

    end = time.time()
    logger.info('Finished Run, Time Elapsed {0} Minutes'.format(round((end-start)/60, 2)))
