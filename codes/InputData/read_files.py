
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

from pathlib import Path
import numpy as np


def read_MSR(mainpath, input_all_n, class_all, set=0):

    '''
        Read MSRAction3D datasets of 3D actions generate by Kinect type sensor.
        '''

    input_all = []

    nseq = 0
    for na in range(10):
        for ns in range(10):
            for ne in range(3):

                path = get_MSR_filename(na, ns, ne, mainpath, set=set)

                file = Path(path)
                if file.is_file():
                    f = open(path, 'r')
                    lines = f.readlines()  # frames
                    f.close()
                    info = np.array([nseq, ns, na, ne])
                    nseq += 1

                    data_all = np.zeros((len(lines), 80))
                    data_all_n = np.zeros((len(lines), 60))
                    cnt_1_zero = 0
                    for i in range(len(lines)):  # number of frames per sequence
                        var = []
                        k = 0
                        for j in range(len(lines[i])):  # number of characters per frame
                            if lines[i][j] != '\t' and lines[i][j] != '\n':
                                var += lines[i][j]

                            else:
                                if lines[i][j] != '\n':
                                    str = ''.join(var)
                                    data = float(str)
                                    # print(data)
                                    data_all[i, k] = data
                                    del var
                                    del data
                                    var = []
                                    k += 1
                        cnt_2_zero=0
                        for id in range(20):
                            data_all_n[i, 3 * id + 0] = data_all[i, 4 * id + 0]
                            data_all_n[i, 3 * id + 1] = 0.25 * data_all[i, 4 * id + 2]
                            data_all_n[i, 3 * id + 2] = 400 - data_all[i, 4 * id + 1]
                            if data_all_n[i, 3 * id + 0] == 0. and data_all_n[i, 3 * id + 1] == 0. \
                                    and data_all_n[i, 3 * id + 2] == 400.:
                                cnt_2_zero += 1

                        if cnt_2_zero == 20:
                            cnt_1_zero += 1

                    if cnt_1_zero == len(lines):
                        pass
                    else:
                        # Collection of information for each action sequence (sequence, actor, action, event)
                        class_all.append(info)
                        # Collection of raw sensor information 4 values per joint (dim=80)
                        input_all.append(data_all)
                        # Collection of processed 3D information 3 Cartesian coordinate parameters per joint (dim=60)
                        input_all_n.append(data_all_n)

    return input_all_n, class_all


def get_MSR_filename(na, ns, ne, mainpath, set):

    dataset = "MSRAction3DDataset_{:d}".format(set)

    if na + 1 >= 10 and ns + 1 < 10:

        filename = "a{:d}_s0{:d}_e0{:d}".format(na + 1, ns + 1, ne + 1)

    elif na + 1 < 10 and ns + 1 >= 10:

        filename = "a0{:d}_s{:d}_e0{:d}".format(na + 1, ns + 1, ne + 1)

    elif na + 1 >= 10 and ns + 1 >= 10:

        filename = "a{:d}_s{:d}_e0{:d}".format(na + 1, ns + 1, ne + 1)

    else:

        filename = "a0{:d}_s0{:d}_e0{:d}".format(na + 1, ns + 1, ne + 1)

    if set == 2:
        return mainpath + "/" + dataset + "/" + filename + "_2nd.txt"

    else:
        return mainpath + "/" + dataset + "/" + filename + ".txt"


