from tensorflow.keras.callbacks import Callback
import os

class SWA(Callback):

    def __init__(self, filepath, n_epoch):
        super(SWA, self).__init__()
        if os.path.isdir(filepath):
            filepath = os.path.join(filepath, 'swa.model')
        self.filepath = filepath
        self.n_epoch = n_epoch

    def on_train_begin(self, logs=None):
        self.swa_epoch = self.params['epochs'] - self.n_epoch
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.n_epoch))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch >= self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                                              (epoch - self.swa_epoch) + 1)
        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print(f'Final stochastic averaged weights saved to file. path: {self.filepath}')
