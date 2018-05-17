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
from pprint import pprint
from numpy.random import multivariate_normal


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

        if padding:
            vocabulary['PAD'] = 0
        if filter:
            vocabulary['UNK'] = 1

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

        elif mode == 'embed':

            n_sentences = 0
            max_length = 0

            for sentence in self._train_data.sentences:
                n_sentences += 1
                if len(sentence) > max_length:
                    max_length = len(sentence)

            z = zarr.open(output, 'w')
            z.create_dataset('lang_data', shape=(n_sentences, max_length), chunks=(256, ), dtype='i')

            counter = 0
            for sentence in self._train_data.sentences:
                idx = []
                for word in sentence:
                    try:
                        idx.append(self._train_data[word])
                    except:
                        idx.append(self._train_data['UNK'])
                pad_size = max_length - len(idx)
                pad = [self._train_data['PAD'] for _ in range(pad_size)]
                idx += pad
                z['lang_data'][counter, :] = idx
                counter += 1


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

        self.embedding3 = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding4 = nn.Embedding(self.n_embeddings, self.embedding_dim)

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
        out = out.gather(1, contexts).sum(dim=1)
        out = out.mean()

        mu_ = self.embedding3(central)
        sigma_ = F.softplus(self.embedding4(central))

        kl = torch.log(sigma_) - torch.log(sigma) + 1./2 * (sigma ** 2 + (mu - mu_) ** 2) / sigma_ ** 2 - 1./2
        kl = kl.sum(dim=1)
        kl = kl.mean()
        loss = kl - out

        return loss

    def train_network(self, data_dir, epochs, batch_size, weight_decay, lr, output_dir):

        self.cuda()
        self.train(True)

        data = zarr.open(data_dir, 'r')

        central_data = data['train_data_central']
        context_data = data['train_data_contexts']

        n_batches = round(central_data.shape[0]/batch_size)

        opt = optim.Adam(self.parameters(), weight_decay=weight_decay, lr=lr)
        loss_f = nn.KLDivLoss()
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

        losses = []

        for epoch in range(epochs):

            start = time.time()
            scheduler.step()
            avg_loss = numpy.zeros((1,))
            n_samples = 1

            for idx in range(n_batches):

                opt.zero_grad()

                central_batch = central_data[idx*batch_size:idx*batch_size+batch_size]
                context_batch = context_data[idx * batch_size:idx * batch_size + batch_size, :]

                central_batch = Variable(torch.LongTensor(central_batch)).cuda()
                context_batch = Variable(torch.LongTensor(context_batch)).cuda()
                noise = multivariate_normal(numpy.zeros(self.embedding_dim), numpy.eye(self.embedding_dim),
                                            [central_batch.size()[0]])
                noise = Variable(torch.FloatTensor(noise), requires_grad=False).cuda()

                loss = self(central_batch, context_batch, noise)
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


class EmbedAlign(nn.Module):

    def __init__(self, vocab_size1, vocab_size2, emb_dimensions):

        super(EmbedAlign, self).__init__()

        self.embedding_dims = emb_dimensions
        self.embeddings = nn.Embedding(vocab_size1, emb_dimensions)
        self.bilstm = nn.LSTM(emb_dimensions, emb_dimensions, bidirectional=True)
        self.linear0 = nn.Linear(emb_dimensions, emb_dimensions)
        self.linear1 = nn.Linear(emb_dimensions, emb_dimensions)
        self.linear2 = nn.Linear(emb_dimensions, emb_dimensions)
        self.linear3 = nn.Linear(emb_dimensions, emb_dimensions)

        self.linear4 = nn.Linear(emb_dimensions, emb_dimensions)
        self.linear5 = nn.Linear(emb_dimensions, vocab_size1)
        self.linear6 = nn.Linear(emb_dimensions, emb_dimensions)
        self.linear7 = nn.Linear(emb_dimensions, vocab_size2)

    def forward(self, en_batch, fr_batch, noise):

        sent_embeddings = self.embeddings(en_batch)
        n_batch, n_words, n_dim = sent_embeddings.shape

        o, i = self.bilstm(sent_embeddings.t())

        o = (o[:, :, :self.embedding_dims] + o[:, :, :self.embedding_dims]).t()

        mu = F.relu(self.linear0(o))
        mu = self.linear1(mu)

        sigma = F.relu(self.linear2(o))
        sigma = F.softplus(self.linear3(sigma))

        z = mu + noise * sigma
        mask = torch.sign(en_batch).float()

        x = F.relu(self.linear4(z))
        x = F.log_softmax(self.linear5(x), dim=-1)
        x = torch.sum(torch.gather(x, 2, en_batch.view(n_batch, n_words, 1)), dim=1)
        x = torch.mean(x)

        y = F.relu(self.linear6(z))
        y = F.log_softmax(self.linear7(y), dim=-1)
        y = torch.mean(torch.gather(y, 2, fr_batch.expand(n_words, -1, -1).t()), dim=1)
        y = torch.sum(y, dim=-1)
        y = torch.mean(y)

        kl = torch.sum(-1./2 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2, dim=-1), dim=-1)
        kl = torch.mean(kl)

        loss = -x - y + kl

        return loss

    def train_network(self, eng_dir, fr_dir, epochs, batch_size, weight_decay, lr, output_dir):

        self.cuda()
        self.train(True)

        opt = optim.Adam(self.parameters(), weight_decay=weight_decay, lr=lr)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

        losses = []

        z1 = zarr.open(eng_dir, 'r')
        eng = z1['lang_data']

        z2 = zarr.open(fr_dir, 'r')
        fr = z2['lang_data']

        n_batches = round(len(eng) / batch_size)

        for epoch in range(epochs):

            start = time.time()
            scheduler.step()
            avg_loss = numpy.zeros((1,))
            n_samples = 1

            for idx in range(n_batches):

                opt.zero_grad()

                eng_batch = eng[idx*batch_size:idx*batch_size+batch_size, :]
                fr_batch = fr[idx*batch_size:idx*batch_size+batch_size, :]

                eng_batch = Variable(torch.LongTensor(eng_batch)).cuda()
                fr_batch = Variable(torch.LongTensor(fr_batch)).cuda()
                noise = multivariate_normal(numpy.zeros(self.embedding_dims), numpy.eye(self.embedding_dims),
                                            [eng_batch.size()[0], eng_batch.size()[1]])
                noise = Variable(torch.FloatTensor(noise), requires_grad=False).cuda()

                loss = self(eng_batch, fr_batch, noise)
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
                              embedding_dim=self.embedding_dims,
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
                    phrase_vec = torch.FloatTensor([0 for _ in range(self.embedding_dims)])
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
                            precision_at += found / (idx + 1)
                            found += 1
                    gap = precision_at / total_weight
                    gaps.append(gap)
                    total_average_gap += gap
                    counter += 1

        Plotter.plot_evaluation(scores=gaps,
                                n_sentences=len(gaps),
                                output=output_dir)

        total_average_gap /= counter
        logger.info('Total Average GAP {0}'.format(total_average_gap))


def run(model, output_dir, epochs, emb_dimensions, lr, batch_size, weight_decay):

    start = time.time()

    if model == 'embed':

        eng_corpus = TrainCorpus(files=['hansards/training.en'], padding=True, filter=10000)
        fr_corpus = TrainCorpus(files=['hansards/training.fr'], padding=True, filter=10000)
        test_corpus = TestCorpus(candidate_file='eval/lst.gold.candidates',
                                 truth_file='eval/lst_test.gold')
        e_featurizer = Featurizer(eng_corpus, test_corpus)
        f_featurizer = Featurizer(fr_corpus, test_corpus)
        e_featurizer.context_words2features(mode='embed',
                                            output=output_dir + '/' + 'e_data.zarr')
        f_featurizer.context_words2features(mode='embed',
                                            output=output_dir + '/' + 'f_data.zarr')
        embed_align = EmbedAlign(vocab_size1=len(eng_corpus.vocabulary),
                                 vocab_size2=len(fr_corpus.vocabulary),
                                 emb_dimensions=emb_dimensions)
        embed_align.train_network(eng_dir=output_dir + '/' + 'e_data.zarr',
                                  fr_dir=output_dir + '/' + 'f_data.zarr',
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  lr=lr,
                                  weight_decay=weight_decay,
                                  output_dir=output_dir)
        embed_align.evaluate(train_corpus=eng_corpus,
                             test_corpus=test_corpus,
                             output_dir=output_dir)

    elif model == 'skipgram':

        train_corpus = TrainCorpus(files=['hansards/training.en'], padding=True, filter=None)
        test_corpus = TestCorpus(candidate_file='eval/lst.gold.candidates',
                                 truth_file='eval/lst_test.gold')
        featurizer = Featurizer(train_corpus, test_corpus)
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
        skipgram.evaluate(train_corpus=train_corpus,
                          test_corpus=test_corpus,
                          output_dir=output_dir)

    elif model == 'bayes':

        train_corpus = TrainCorpus(files=['hansards/training.en'], padding=True, filter=None)
        test_corpus = TestCorpus(candidate_file='eval/lst.gold.candidates',
                                 truth_file='eval/lst_test.gold')
        featurizer = Featurizer(train_corpus, test_corpus)
        featurizer.context_words2features(mode='bayes',
                                          output=output_dir + '/' + 'data.zarr')
        bayes_skipgram = BayesSkipgram(n_layers=3,
                                       n_features=len(train_corpus.vocabulary),
                                       corpus=train_corpus,
                                       embedding_dim=emb_dimensions)
        bayes_skipgram.train_network(data_dir=output_dir + '/' + 'data.zarr',
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     lr=lr,
                                     weight_decay=weight_decay,
                                     output_dir=output_dir)
        bayes_skipgram.evaluate(train_corpus=train_corpus,
                                test_corpus=test_corpus,
                                output_dir=output_dir)

    else:
        raise NotImplementedError

    end = time.time()
    logger.info('Finished Run, Time Elapsed {0} Minutes'.format(round((end - start) / 60, 2)))


if __name__ == '__main__':

    #skipgram, bayes, embed
    run(model='skipgram',
        output_dir='/home/oem/PycharmProjects/ULLProject2',
        epochs=10,
        emb_dimensions=300,
        lr=0.005,
        batch_size=256,
        weight_decay=0.0001)