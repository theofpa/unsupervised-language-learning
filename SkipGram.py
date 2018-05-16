from whoosh.analysis import StandardAnalyzer
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import time
import numpy
import funcy
import zarr
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter

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

    def __init__(self, files):

        Corpus.__init__(self, files)

        self._preprocessor = StandardAnalyzer()
        self._vocabulary = self._build_vocabulary()
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

    def _build_vocabulary(self):

        logger.info('Building Vocabulary, Sentences')

        words = []

        for raw_content in self._content:
            content = raw_content.split('\n')
            words.extend([token.text for token in self._preprocessor(raw_content, removestops=False)])

        vocabulary = {}
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

        return vocabulary

    @property
    def contexts(self, window_size=2):

        #logger.info('Building Contexts, Window Size {0}'.format(window_size))

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


class TestCorpus(Corpus):

    def __init__(self, candidate_file, truth_file):

        Corpus.__init__(self, files=[candidate_file, truth_file])

        self._preprocessor = StandardAnalyzer()
        self.candidates = self._load_candidates(candidate_file)
        self.ground_truth = self._load_truth(truth_file)

    def __str__(self):
        return 'Corpus, {candidates!s} candidates, {truth!s} truth'.format(candidates=len(self.candidates),
                                                                                  truth=len(self.ground_truth))

    def __repr__(self):
        return 'Class {name!r}'.format(name=__class__.__name__)

    def _read(self, files):
        content = []
        for file in files:
            logger.info('Reading File {0}'.format(file))
            with open(file, 'r') as f:
                content.append(f.read())
        return content

    def _load_candidates(self, candidate_file):
        with open(candidate_file, 'r') as f:
            tar_c = {}
            for line in f:
                line = line.strip().split('::')
                target = line[0].split('.')[0]
                candidates = line[1].split(';')
                for idx in range(len(candidates)):
                    candidates[idx] = candidates[idx].split()
                tar_c[target] = candidates
        return tar_c

    def _load_truth(self, truth_file):
        with open(truth_file, 'r') as f:
            tr_c = {}
            for line in f:
                line = line.strip().split('::')
                target = line[0].split()[0].split('.')[0]
                truth = list(filter(None, line[1].strip().split(';')))
                for idx in range(0, len(truth)):
                    phrase = truth[idx].split()[0:-1]
                    weight =truth[idx].split()[-1]
                    pair = tuple([phrase, weight])
                    truth[idx] = pair
                if target not in tr_c:
                    tr_c[target] = []
                    tr_c[target].append(truth)
                else:
                    tr_c[target].append(truth)
        return tr_c


class Featurizer:

    def __init__(self, train_corpus, test_corpus):

        self._train_data = train_corpus
        self._test_data = test_corpus

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    def context_words2features(self, mode, output):

        logger.info('Building Training Data, Labels from Contexts')

        word_contexts = self._train_data.contexts
        n_context_words = self._train_data._n_context_words
        n_features = len(self._train_data.vocabulary)
        window_size = self._train_data._window_size

        if mode == 'normal':

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

        elif mode == 'bayes':

            window_size *= 2

            z = zarr.open(output, 'w')
            z.create_dataset('train_data_central', shape=(n_context_words, ), chunks=(256, ), dtype='i')
            z.create_dataset('labels', shape=(n_context_words, ), chunks=(256,), dtype='i')
            z.create_dataset('train_data_contexts', shape=(n_context_words, window_size), chunks=(256,), dtype='i')

            counter = 0

            for word, contexts in word_contexts.items():

                logger.info('Processing word {word}, contexts {size}'.format(word=word, size=len(contexts)))

                word_idx = self._train_data[word]
                train_data_central = []
                context_idx = []

                for context in contexts:
                    for context_word in context:
                        train_data_central.append(word_idx)
                        context_idx.append(self._train_data[context_word])

                n_words = len(contexts) * window_size

                z['train_data_central'][counter:counter + n_words] = numpy.array(train_data_central, dtype=numpy.int32)
                z['labels'][counter:counter+n_words] = numpy.array(context_idx, dtype=numpy.int32)
                context_idx = funcy.partition(window_size, context_idx)
                context_idx = [context for context in context_idx for _ in range(window_size)]
                z['train_data_contexts'][counter:counter + n_words, :] = numpy.array(context_idx, dtype=numpy.int32)

                counter += n_words


class Plotter:

    @staticmethod
    def plot_training(epochs, losses, embedding_dim, output):
        plt.figure()
        plt.title('WordEmbeddings')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot([i for i in range(epochs)], losses, 'r', label='WordEmbeddings %d' % embedding_dim)
        plt.legend()
        plt.grid(True)
        plt.savefig(output + '/' + 'training.png')
        plt.show()

    @staticmethod
    def plot_evaluation(scores, n_sentences, output):
        plt.figure()
        plt.title('GapScores')
        plt.xlabel('Sentence')
        plt.ylabel('Score')
        plt.plot([i for i in range(n_sentences)], scores, 'b', label='GapScores')
        plt.legend()
        plt.grid(True)
        plt.savefig(output + '/' + 'eval.png')
        plt.show()


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

    def train_network(self, data_dir, epochs, batch_size, weight_decay, lr, momentum, output_dir):

        self.cuda()
        self.train(True)

        data = zarr.open(data_dir, 'r')

        train_data = data['train_data']
        labels = data['labels']

        n_batches = round(train_data.shape[0]/batch_size)
        opt = optim.SGD(self.parameters(), weight_decay=weight_decay, lr=lr, momentum=momentum)
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

    def evaluate(self, train_corpus, test_corpus, output_dir):

        self.train(False)

        embeddings = list(self.parameters())[0]

        candidates = test_corpus.candidates
        truth = test_corpus.ground_truth
        ranked = {}

        for target, candidate in candidates.items():
            try:
                idx = train_corpus[target]
                target_vec = embeddings[idx, :].cpu().data.numpy()
            except:
                logger.warning('Target out of Vocabulary {0}'.format(target))
            else:
                ranking = []
                for phrase in candidate:
                    n_words = 0
                    phrase_vec = torch.FloatTensor([0 for _ in range(self.n_hidden)])
                    for word in phrase:
                        try:
                            idx = train_corpus[word]
                        except:
                            logger.warning('Candidate out of Vocabulary {0}'.format(word))
                        else:
                            n_words += 1
                            phrase_vec += embeddings[idx, :].cpu().data

                    if n_words:
                        phrase_vec /= len(phrase)

                    phrase_vec = phrase_vec.numpy()
                    sim = cosine_similarity(target_vec.reshape(1, -1), phrase_vec.reshape(1, -1))[0][0]
                    ranking.append(tuple([phrase, sim]))
                ranking = sorted(ranking, key=itemgetter(1), reverse=True)
                ranking = [i[0] for i in ranking]
                ranked[target] = ranking

        total_average_gap = 0
        counter = 1
        gaps = []

        for target, sentences in truth.items():
            try:
                ranking = ranked[target]
            except:
                pass
            else:
                for sentence in sentences:
                    total_weight = sum([int(i[1]) for i in sentence])
                    tokens = [i[0] for i in sentence]
                    found = 1
                    precision_at = 0
                    for idx in range(len(ranking)):
                        if ranking[idx] in tokens:
                            precision_at += found/(idx+1)
                            found += 1
                    gap = precision_at/total_weight
                    gaps.append(gap)
                    total_average_gap += gap
                    counter += 1

        Plotter.plot_evaluation(scores=gaps,
                                n_sentences=len(gaps),
                                output=output_dir)

        total_average_gap /= counter
        logger.info('Total Average GAP {0}'.format(total_average_gap))


class BayesSkipgram(nn.Module):

    def __init__(self, n_features, n_layers, corpus, embedding_dim):

        super(BayesSkipgram, self).__init__()

        self.n_layers = n_layers
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.window_size = corpus.window_size * 2
        self.n_embeddings = len(corpus.vocabulary)

        self.embedding1 = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding2 = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.linear1 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.linear2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear3 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear4 = nn.Linear(self.embedding_dim, self.n_embeddings)

    def forward(self, central, contexts, noise):

        x = self.embedding1(central)
        y = self.embedding2(contexts)

        x = x.view((x.size()[0], 1, self.embedding_dim))
        x = torch.cat([x for _ in range(self.window_size)], dim=1)
        y = torch.cat([y, x], dim=2)

        out = None
        for idx in range(self.window_size):
            if out is None:
                out = F.relu(self.linear1(y[:, idx, :]))
            else:
                out += F.relu(self.linear1(y[:, idx, :]))

        sigma = F.softplus(self.linear2(out))
        mu = self.linear3(out)
        z = mu + noise*sigma

        out = F.log_softmax(self.linear4(out), dim=1)
        #out = F.softmax(self.linear4(z), dim=1)

        return out

    def train_network(self, central_data, context_data, labels, epochs, batch_size, weight_decay, lr, momentum, output_dir):

        self.cuda()
        self.train(True)

        n_batches = round(central_data.shape[0]/batch_size)

        opt = optim.SGD(self.parameters(), weight_decay=weight_decay, lr=lr, momentum=momentum)
        loss_f = nn.KLDivLoss()
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

        losses = []

        for epoch in range(epochs):

            start = time.time()
            scheduler.step()
            avg_loss = numpy.zeros((1,))
            n_samples = 1

            for idx in range(0, n_batches):

                opt.zero_grad()

                central_batch = central_data[idx*batch_size:idx*batch_size+batch_size]
                context_batch = context_data[idx * batch_size:idx * batch_size + batch_size, :]
                label_batch = labels[idx*batch_size:idx*batch_size+batch_size]
                label_batch = self.vocabulary2one_hot(label_batch)
                noise = torch.randn(1, self.embedding_dim)

                central_batch = Variable(torch.LongTensor(central_batch)).cuda()
                context_batch = Variable(torch.LongTensor(context_batch)).cuda()
                noise = Variable(noise, requires_grad=False).cuda()
                label_batch = Variable(torch.FloatTensor(label_batch), requires_grad=False).cuda()

                output = self(central_batch, context_batch, noise)
                loss = loss_f(output, label_batch)
                loss.backward()
                opt.step()
                n_samples += 1
                avg_loss += loss.cpu().data.numpy()

            end = time.time()
            avg_loss /= n_samples
            losses.append(avg_loss)
            logger.info('Epoch {0}, Average Loss {1}, Minutes spent {2}'
                    .format(epoch + 1, round(avg_loss.data[0], 4), round((end-start)/60., 2)))

        Plotter.plot_training(epochs=epochs,
                              losses=losses,
                              embedding_dim=self.embedding_dim,
                              output=output_dir)

    def vocabulary2one_hot(self, idx):

        size = idx.shape[0]
        tensor = scipy.sparse.coo_matrix(([1 for _ in range(size)], ([i for i in range(size)], list(idx))), shape=(size, self.n_embeddings))
        return tensor.toarray()

    def evaluate(self, train_corpus, test_corpus, output_dir):

        self.train(False)

        embeddings = list(self.parameters())[0]

        candidates = test_corpus.candidates
        truth = test_corpus.ground_truth
        ranked = {}

        for target, candidate in candidates.items():
            try:
                idx = train_corpus[target]
                target_vec = embeddings[idx, :].cpu().data.numpy()
            except:
                logger.warning('Target out of Vocabulary {0}'.format(target))
            else:
                ranking = []
                for phrase in candidate:
                    n_words = 0
                    phrase_vec = torch.FloatTensor([0 for _ in range(self.embedding_dim)])
                    for word in phrase:
                        try:
                            idx = train_corpus[word]
                        except:
                            logger.warning('Candidate out of Vocabulary {0}'.format(word))
                        else:
                            n_words += 1
                            phrase_vec += embeddings[idx, :].cpu().data

                    if n_words:
                        phrase_vec /= len(phrase)

                    phrase_vec = phrase_vec.numpy()
                    sim = cosine_similarity(target_vec.reshape(1, -1), phrase_vec.reshape(1, -1))[0][0]
                    ranking.append(tuple([phrase, sim]))
                ranking = sorted(ranking, key=itemgetter(1), reverse=True)
                ranking = [i[0] for i in ranking]
                ranked[target] = ranking

        total_average_gap = 0
        counter = 1
        gaps = []

        for target, sentences in truth.items():
            try:
                ranking = ranked[target]
            except:
                pass
            else:
                for sentence in sentences:
                    total_weight = sum([int(i[1]) for i in sentence])
                    tokens = [i[0] for i in sentence]
                    found = 1
                    precision_at = 0
                    for idx in range(len(ranking)):
                        if ranking[idx] in tokens:
                            precision_at += found/(idx+1)
                            found += 1
                    gap = precision_at/total_weight
                    gaps.append(gap)
                    total_average_gap += gap
                    counter += 1

        Plotter.plot_evaluation(scores=gaps,
                                n_sentences=len(gaps),
                                output=output_dir)

        total_average_gap /= counter
        logger.info('Total Average GAP {0}'.format(total_average_gap))


if __name__ == '__main__':

    start = time.time()
    train_corpus = TrainCorpus(files=['hansards/training.en'])

    test_corpus = TestCorpus(candidate_file='eval/lst.gold.candidates',
                             truth_file='eval/lst_test.gold')


    featurizer = Featurizer(train_corpus, test_corpus)

###################

    featurizer.context_words2features(mode='normal',
                                      output='/home/oem/PycharmProjects/ULLProject2/data.zarr')

    skipgram = Skipgram(n_layers=3,
                      n_hidden=100,
                      n_features = len(train_corpus.vocabulary))

    skipgram.train_network(data_dir='/home/oem/PycharmProjects/ULLProject2/data.zarr',
                          epochs=10,
                          batch_size=256,
                          weight_decay=0.0001,
                          lr=0.005,
                          momentum=0.9,
                          output_dir='/home/oem/PycharmProjects/ULLProject2')

    skipgram.evaluate(train_corpus=train_corpus,
                      test_corpus=test_corpus,
                      output_dir='/home/oem/PycharmProjects/ULLProject2')

###################

    #featurizer.context_words2features(mode='bayes',
    #                                  output='/home/oem/PycharmProjects/ULLProject2/data.zarr')
#
    #m = zarr.open('/home/oem/PycharmProjects/ULLProject2/data.zarr', 'r')
#
    #central_words = m['train_data_central']
    #contexts = m['train_data_contexts']
    #bay_labels = m['labels']
#
    #bayes_skipgram = BayesSkipgram(n_layers=3,
    #                  n_features=len(train_corpus.vocabulary),
    #                  corpus=train_corpus,
    #                  embedding_dim=100)
#
    #bayes_skipgram.train_network(central_data=central_words,
    #                       context_data=contexts,
    #                       labels=bay_labels,
    #                       epochs=10,
    #                       batch_size=256,
    #                       lr=0.005,
    #                       weight_decay=0.0001,
    #                       momentum=0.9)
#
    #bayes_skipgram.evaluate(train_corpus, test_corpus)

###################
    end = time.time()
    logger.info('Finished Run, Time Elapsed {0} Minutes'.format(round((end-start)/60, 2)))
