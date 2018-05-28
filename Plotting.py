import matplotlib.pyplot as plt


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