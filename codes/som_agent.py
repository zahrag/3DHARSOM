
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
from SOM import SOM
from SNN import SNN


class somagent_phase_I:

    def __init__(self, learning, l_x, l_y, input_size, sigma, softmax_exponent, max_epoch, dyn_as_input):

        self.net_1 = SOM(learning=learning,
                         outputsize_x=l_x,
                         outputsize_y=l_y,
                         inputsize=input_size,
                         sigma=sigma,
                         softmax_exponent=softmax_exponent,
                         max_epoch=max_epoch)

        self.dyn_as_input = dyn_as_input

    def run(self, data, data_d1, data_d2, data_index, learning=None):

        self.net_1.learning = learning

        all_activity_pattern = []
        iteration = 0
        epoch = 0
        run = True
        while run:

            epoch += 1

            # Random selection
            rseq = np.random.permutation(len(data_index))
            all_activity_pattern = []
            for nseq in range(len(data_index)):  # Sequences

                if learning is False:
                    ind_seq = int(data_index[nseq])
                else:
                    ind_seq = int(data_index[rseq[nseq]])

                data_seq_d0 = data[ind_seq]
                data_seq_d1 = data_d1[ind_seq]
                data_seq_d2 = data_d2[ind_seq]
                if self.dyn_as_input == 1:
                    data_seq_d0 = np.concatenate((data_seq_d0, data_seq_d1), axis=1)

                elif self.dyn_as_input == 2:
                    data_seq_d0 = np.concatenate((data_seq_d0, data_seq_d1), axis=1)
                    data_seq_d0 = np.concatenate((data_seq_d0, data_seq_d2), axis=1)

                activity_pattern = np.zeros((np.size(data_seq_d0, 0), 2))
                for nfr in range(np.size(data_seq_d0, 0)):  # Frames per sequence
                    iteration += 1

                    # running first-layer SOM
                    # print('input dim phase_I:', len(data_seq_d0[nfr, :]))
                    activity, winner = self.net_1.run_SOM(data_seq_d0[nfr, :])
                    # print("\nInput:\n", data_seq_d0[nfr, :])
                    # print("\nWinner:", winner)
                    # print("\nmin_actinity:", np.min(activity), "\t max_actinity:", np.max(activity))
                    activity_pattern[nfr, 0] = winner[0]
                    activity_pattern[nfr, 1] = winner[1]

                all_activity_pattern.append(activity_pattern)

            if learning:
                print("", end='\r')
                print("Phase:{}  \t Epoch:{} \t Row:{} \t Column:{}".format(2, epoch, np.size(self.net_1.weights, 0),
                                                                            np.size(self.net_1.weights, 1)), end="", flush=True)
            if epoch == self.net_1.max_epoch or learning is False:
                run = False

        return all_activity_pattern


class somagent_phase_II:

    def __init__(self, learning, l_x, l_y, input_size, sigma, softmax_exponent, max_epoch, class_number):

        self.net_2 = SOM(learning=learning,
                         outputsize_x=l_x,
                         outputsize_y=l_y,
                         inputsize=input_size,
                         sigma=sigma,
                         softmax_exponent=softmax_exponent,
                         max_epoch=max_epoch)

        self.net_3 = SNN(learning=learning,
                         outputsize_x=class_number,
                         outputsize_y=1,
                         inputsize=l_x*l_y)

    def run(self, data, data_index, data_class_info, learning=None):

        self.net_2.learning = learning
        self.net_3.learning = learning
        # Performance results
        result_per_class = np.zeros((1, self.net_3.outputsize_x))
        snn_activity_map = []
        epoch = 0
        run = True
        while run:

            epoch += 1

            rseq = np.random.permutation(len(data_index))
            t_res = 0
            for nseq in range(len(data_index)):

                if learning is False:
                    ind_seq = int(data_index[nseq])
                else:
                    ind_seq = int(data_index[rseq[nseq]])

                # class label
                class_seq = data_class_info[ind_seq]
                # print('input dim phase_II:', len(data[ind_seq]))
                activity, winner = self.net_2.run_SOM(data[ind_seq])
                snn_activity, snn_result = self.net_3.run_SNN(activity.flatten(), int(class_seq[2]))
                result_per_class[0, int(class_seq[2])] += snn_result
                t_res += snn_result

                # get third-layer activation maps for Test sequences
                if learning is False:
                    snn_activity_map.append(snn_activity.T)

            if learning:
                print("", end='\r')
                print("Phase:{}  \t Epoch:{} \t Row:{} \t Column:{} \t Result:{}".format(2, epoch,
                                                                                         np.size(self.net_2.weights, 0),
                                                                                         np.size(self.net_2.weights, 1),
                                                                                         100*t_res/len(data_index)), end="", flush=True)
            if epoch == self.net_2.max_epoch or learning is False:
                run = False

        return result_per_class, snn_activity_map


