import dask
import dask.array
import zarr
import xarray
import numpy
import logging
from pprint import pprint
import time
import multiprocessing
from dask.distributed import Client
from dask.multiprocessing import get
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
logger = logging.getLogger('ULL')
logging.basicConfig(level=logging.INFO)


class Manager:

    def __init__(self):
        pass

    @staticmethod
    def data2zarr(file, output):
        with open(file) as f:
            vocabulary, word_vectors = [], []
            for line in f:
                word = line.split()[0]
                word_vector = line.split()[1:]
                vocabulary.append(word)
                word_vectors.append(word_vector)
            word_vectors = numpy.array(word_vectors, dtype=numpy.float32)
            root = zarr.open(output, mode='w')
            root.create_dataset('word_vectors', data=word_vectors, chunks=(1, 300))
            root.create_dataset('words', data=vocabulary, dtype=str)

    @staticmethod
    def truth2zarr(file, output):
        with open(file) as f:
            sample1, sample2 , similarities = [], [], []
            next(f)
            for line in f:
                word1, word2, similarity = line.split()[0], line.split()[1], line.split()[2]
                sample1.append(word1)
                sample2.append(word2)
                similarities.append(similarity)
            similarities = numpy.array(similarities, dtype=numpy.float32)
            root = zarr.open(output, mode='w')
            root.create_dataset('similarities', data=similarities)
            root.create_dataset('sample1', data=sample1, dtype=str)
            root.create_dataset('sample2', data=sample2, dtype=str)


class Analyzer:

    def __init__(self):
        pass

    def load_test_data(self, file):
        data = zarr.open(file, 'r')
        samples1 = list(data['sample1'][:])
        samples2 = list(data['sample2'][:])
        similarities = list(data['similarities'][:])

        return samples1, samples2, similarities, file

    def load_train_data(self, file):
        data = zarr.open(file, 'r')
        word_vectors = dask.array.from_array(data['word_vectors'], chunks=(1, 300))
        vocabulary = list(data['words'][:])
        train_data = xarray.DataArray(data=word_vectors,
                                    dims=('word', 'dimension'),
                                    coords={'word': vocabulary})

        return train_data, file

    def cossim(self, train_data, test_data):
        samples1, samples2, similarities, test_file = test_data
        data_array, train_file = train_data
        vocabulary = data_array.coords['word'].data
        oov = []
        for idx in range(len(samples1)):
            if samples1[idx] not in vocabulary or samples2[idx] not in vocabulary:
                oov.append(idx)
        for idx in oov:
            del samples1[idx]
            del samples2[idx]
            del similarities[idx]

        word_vectors1 = data_array.sel(word=samples1).data.compute()
        word_vectors2 = data_array.sel(word=samples2).data.compute()
        cossim = cosine_similarity(word_vectors1, word_vectors2).diagonal()
        return cossim, similarities, train_file, test_file

    def stats(self, data):
        cossim, similarities, train_file, test_file = data
        spearman = spearmanr(cossim, similarities)
        pearson = pearsonr(cossim, similarities)
        logger.info('Spearman - {0}, Pearson - {1}, Model {2}, {3}, {4}'
                    .format(spearman, pearson, train_file, test_file, multiprocessing.current_process().name))


if __name__ == '__main__':

    client = Client()

    tasks = {
            #'bow2words': (Manager.data2zarr, 'bow2.words', 'bow2model.zarr'),
            #'bow5words': (Manager.data2zarr, 'bow5.words', 'bow5model.zarr'),
            #'depswords': (Manager.data2zarr,  'deps.words', 'dep2model.zarr'),
            #'simlex': (Manager.truth2zarr, 'SimLex-999/SimLex-999.txt', 'SimLex.zarr'),
            #'men': (Manager.truth2zarr, 'MEN/MEN_dataset_natural_form_full', 'Men.zarr'),
            'analyzer': Analyzer(),
            'load_bow2model': (Analyzer.load_train_data, 'analyzer', 'bow2model.zarr'),
            'load_bow5model': (Analyzer.load_train_data, 'analyzer', 'bow5model.zarr'),
            'load_depsmodel': (Analyzer.load_train_data, 'analyzer', 'dep2model.zarr'),
            'load_simlex': (Analyzer.load_test_data, 'analyzer', 'SimLex.zarr'),
            'load_men': (Analyzer.load_test_data, 'analyzer', 'Men.zarr'),
            'simlex_bow2_cossim': (Analyzer.cossim, 'analyzer', 'load_bow2model', 'load_simlex'),
            'simlex_bow5_cossim': (Analyzer.cossim, 'analyzer', 'load_bow5model', 'load_simlex'),
            'simlex_deps_cossim': (Analyzer.cossim, 'analyzer', 'load_depsmodel', 'load_simlex'),
            'men_bow2_cossim': (Analyzer.cossim, 'analyzer', 'load_bow2model', 'load_men'),
            'men_bow5_cossim': (Analyzer.cossim, 'analyzer', 'load_bow5model', 'load_men'),
            'men_deps_cossim': (Analyzer.cossim, 'analyzer', 'load_depsmodel', 'load_men'),
            'simlex_bow2_stats': (Analyzer.stats, 'analyzer', 'simlex_bow2_cossim'),
            'simlex_bow5_stats': (Analyzer.stats, 'analyzer', 'simlex_bow5_cossim'),
            'simlex_deps_stats': (Analyzer.stats, 'analyzer', 'simlex_deps_cossim'),
            'men_bow2_stats': (Analyzer.stats, 'analyzer', 'men_bow2_cossim'),
            'men_bow5_stats': (Analyzer.stats, 'analyzer', 'men_bow5_cossim'),
            'men_deps_stats': (Analyzer.stats, 'analyzer', 'men_deps_cossim')
    }
    start = time.time()
    #client.get(tasks, 'men')
    client.get(tasks, ['simlex_bow2_stats', 'simlex_bow5_stats', 'simlex_deps_stats',
                       'men_bow2_stats', 'men_bow5_stats', 'men_deps_stats'])
    end = time.time()
    logger.info('Time Elapsed {0} Seconds'.format(round(end-start, 2)))
