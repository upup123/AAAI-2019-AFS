import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FixedLocator
class BatchCreate(object):
    def __init__(self,images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]
    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch
        '''
        Disruption in the first epoch
        '''
        if self._epochs_completed ==0 and start ==0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
        if start+batch_size>self._num_examples:
            #finished epoch
            self._epochs_completed += 1
            '''
            When the remaining sample number of an epoch is less than batch size,
            the difference between them is calculated.
            '''
            rest_num_examples = self._num_examples-start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            '''Disrupt the data'''
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            '''next epoch'''
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part),axis=0),np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
def show_result(ac_score_list,dataset_name):
    plot_spc = [5, 10]
    left_spc = 1
    len_params = 295

    use_arr = ac_score_list[left_spc:int((len_params - plot_spc[0]) / plot_spc[1])]
    ax = plt.subplot()
    ax.plot(np.arange(plot_spc[0] + plot_spc[1] * left_spc, len_params, plot_spc[1]), use_arr, '-o',
            label='AFS')
    plt.title(dataset_name)
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.ylim(0.0, 1.0)
    xmajorLocator = FixedLocator([15, 85, 155, 225, 295])
    xminorLocator = MultipleLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.grid(True, which='minor')

    ymajorLocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.grid(True, which='major')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    plt.show()