import tensorflow as tf

class TFMemory():

    def __init__(self, buffer_size):
        self.MEMORY_KEYS = ('observation', 'action', 'reward', 'done', 'next_observation')
        self.buffer_size = buffer_size
        self.data = {key:[] for key in self.MEMORY_KEYS}

    def remember(self, **kwargs):
        for key in kwargs:
            prev = self.data[key]
            new = tf.expand_dims(kwargs[key], axis=0)
            if len(prev) == 0:
                self.data[key] = new
            else:
                self.data[key] = tf.concat((prev, new), axis=0)[-self.buffer_size:] #pylint: disable=all

    def sample(self, size:int, method='random'):
        data_len = len(self.data['observation'])

        if method == 'random':
            indices = tf.random.shuffle(tf.range(data_len))[:size]
            return [tf.gather(self.data[key], indices) for key in self.MEMORY_KEYS]
        elif method == 'last':
            return [self.data[key][-size:] for key in self.MEMORY_KEYS]
        else:
            raise NotImplementedError
