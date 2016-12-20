import numpy as np
import numpy.random as npr
import chainer as ch

def convert_sequence(sequence, conversion_func):
    return [ conversion_func(s) for s in sequence ]

def convert_sequences(sequences, conversion_func):
    return [ convert_sequence(sequence, conversion_func) for sequence in sequences ]

def sequences2arrays(sequences, dtype=np.int32):
    return ch.functions.transpose_sequence([ np.array(seq, dtype=dtype) for seq in sequences])

class SequenceIterator(ch.dataset.iterator.Iterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.n = n = len(self.dataset)
        self.batch_size = batch_size
        self._repeat = repeat
        self.shuffle = shuffle
        self.n_batches = n//batch_size if (n % batch_size == 0) else n//batch_size + 1

        if shuffle:
            npr.shuffle(self.dataset)

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        minibatch = self.dataset[i*self.batch_size:(i+1)*self.batch_size]
        # minibatches must be sorted by descending sequence len
        # sort key changes if dataset is one set of sequences or a zipped set of seequences
        if type(self.dataset[0][0]) not in (tuple, list, dict):
            # one dataset
            minibatch.sort(key=lambda x:len(x), reverse=True)
        else:
            # zipped or dict dataset
            minibatch.sort(key=lambda x:len(x[0]), reverse=True)

        # last minibatch in epoch
        if self.current_position == self.n_batches-1:
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
            if self.shuffle:
                npr.shuffle(self.dataset)
        else:
            self.is_new_epoch = False
            self.current_position += 1

        return minibatch

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)

def print_batch_loss(loss, epoch_i, batch_i, n_batches):
    batch_percent = float(batch_i)/n_batches
    progress = "Epoch {0} : [{1}{2}] {3:2.2f}%, Loss = {4:2.6f}".format(epoch_i,
                                                           int(np.floor(batch_percent*10))*'=',
                                                           int(np.ceil((1-batch_percent)*10))*'-',
                                                           batch_percent*100,
                                                           float(loss))
    print '\r',progress,

def print_epoch_loss(epoch_i, avg_loss, valid_loss, time):
    print '\rEpoch {0} : Avg Loss = {1:2.4f}, Validation Loss = {2:2.4f}, {3} sec'.format(
        epoch_i, float(avg_loss), float(valid_loss), np.ceil(int(time)))

def sec2hms(secs):
    "Returns an h:m:s string from some float number of seconds"
    m,s = divmod(secs, 60)
    h,m = divmod(m, 60)
    return '{0:02.0f}:{1:02.0f}:{2:02.0f}'.format(h,m,s)
