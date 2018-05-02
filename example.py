# -------------------------------------------------------------------------------------------------
# Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# An online data generator for contour image generation
#
# Author: Salman Khan
# Date  : 01/05/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.backend as K


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_ids, labels, batch_size=32, dim=(227, 227, 3), shuffle=True):

        """

        :param list_ids:
        :param labels:
        :param batch_size:
        :param dim:
        :param shuffle:
        """
        self.labels = labels
        self.list_ids = list_ids
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle

        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates after each epoch

        :return:
        """
        self.indexes = np.arange(len(self.list_ids))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """

        :param list_ids_temp:
        :return:
        """
        x_arr = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        y_arr = np.zeros(self.batch_size)

        for i, list_id in enumerate(list_ids_temp):
            print("Loading",'data/' + list_id + '.jpg')
            x_arr[i, ] = plt.imread( 'data/' + list_id + '.jpg')
            y_arr[i] = self.labels[list_id]

        return x_arr, y_arr

    def __len__(self):
        """ The number of batches per epoch """
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:
        :return:
        """
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find the list of ids
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate the Data
        x_arr, y_arr = self.__data_generation(list_ids_temp)

        return x_arr, y_arr


if __name__ == '__main__':

    plt.ion()
    K.clear_session()

    params = {
        'dim': (227, 227, 3),
        'batch_size': 10,
        'shuffle': True,
    }

    partition = [
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__0",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__1",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__2",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__3",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__4",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__5",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__6",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__7",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__8",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__9",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__10",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__11",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__12",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__13",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__14",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__15",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__16",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__17",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__18",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__19",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__0",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__1",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__2",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__3",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__4",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__5",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__6",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__7",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__8",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__9",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__10",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__11",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__12",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__13",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__14",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__15",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__16",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__17",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__18",
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__19",
    ]

    labels = {
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__0": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__1": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__2": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__3": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__4": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__5": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__6": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__7": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__8": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__9": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__10": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__11": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__12": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__13": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__14": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__15": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__16": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__17": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__18": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_15/orient_92_clen_9_beta_15__19": 2.15,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__0": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__1": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__2": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__3": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__4": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__5": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__6": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__7": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__8": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__9": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__10": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__11": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__12": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__13": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__14": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__15": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__16": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__17": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__18": 1.90,
        "/curved_contours/filter_5_orient_92/c_len_9/beta_30/orient_92_clen_9_beta_30__19": 1.90,
    }

    train_generator = DataGenerator(
        partition,
        labels,
        batch_size=params['batch_size'],
        dim=params['dim'],
        shuffle=False
    )

    gen_out = iter(train_generator)

    X, y = gen_out.next()

    plt.figure()
    plt.imshow(X[0, ])





