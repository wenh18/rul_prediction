import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
metadata = np.load('ne_data/meta_data.npy', allow_pickle=True)


def filter_out_training_extremes(data, ruls, threshold=0.05, rounds=10):
    for _ in range(rounds):
        dataidx = 1
        while dataidx < data.shape[0]:
            if data[dataidx][0] > data[dataidx - 1][0] + threshold:
                data = np.vstack((data[:dataidx], data[dataidx + 1:]))
                # ruls = np.hstack((ruls[:dataidx], ruls[dataidx + 1:]))
            if data[dataidx][0] < data[dataidx - 1][0] - threshold:
                data = np.vstack((data[:dataidx], data[dataidx + 1:]))
                # ruls = np.hstack((ruls[:dataidx], ruls[dataidx ]))
            dataidx += 1
        ruls = ruls[-data.shape[0]:]
    return data, ruls

def interp(x, y, num, ruls):
    ynew = []
    for i in range(y.shape[1]):
        f = interpolate.interp1d(x, y[:, i], kind='linear')
        x_new = np.linspace(x[0], x[-1], num)
        ytmp = f(x_new)
        ynew.append(ytmp)
    ynew = np.vstack(ynew)
    ynew = ynew.T
    newruls = [i for i in range(1, ynew.shape[0] + 1)]
    newruls.reverse()
    newruls = np.array(newruls).astype(float)
    new_right_end_value = ruls[-1] * (num/len(x))
    for i in range(len(newruls)):
        newruls[i] += new_right_end_value
    return ynew, newruls


namelist = metadata[0] + metadata[1]
for displayidx in range(0, 1):
    for batteryname in namelist:
        name = 'ne_data/' + batteryname + 'v4.npy'
        rul = 'ne_data/' + batteryname + '_rulv4.npy'
        a0 = np.load(name, allow_pickle=True)
        ruls = np.load(rul, allow_pickle=True)
        ratio = 1/1.2
        # print(ruls[-1])
        # newruls = change_ruls(ruls, ratio)
        # print(newruls)
        seqlen = a0.shape[0]

        data, allrul = filter_out_training_extremes(a0, ruls, threshold=0.05)
        print(allrul)
        sohs = data[:, 0]
        newdata, newrul = interp([i for i in range(data.shape[0])], data,
                                 int(ratio*data.shape[0]), allrul)
        print(newrul)
        # if data.shape[0] != allrul.shape[0]:
        #     print('error')
        # else:
        #     print(data.shape, allrul.shape)

        plt.plot([i for i in range(len(data))], data[:, displayidx], )
        plt.plot([i for i in range(len(newdata))], newdata[:, displayidx], )
    # plt.savefig('train.jpg')
    # plt.xlim((0, 100))
    plt.show()

exit(0)

for displayidx in range(0, 1):
    for batteryname in metadata[0] + metadata[1]:
        name = 'ne_data/' + batteryname + 'v3.npy'
        rul = 'ne_data/' + batteryname + '_rulv3.npy'
        a0 = np.load(name, allow_pickle=True)
        ruls = np.load(rul, allow_pickle=True)
        print(ruls)
        seqlen = a0.shape[0]
        # a0 = a0.T
        # if seqlen >= 500:
        #     data = filter_out_training_extremes(a0[:500], threshold=0.05)
        #     plt.plot([i for i in range(len(data))], data[:, 0], )
        # else:
        data, allrul = filter_out_training_extremes(a0, ruls, threshold=0.05)
        if data.shape[0] != allrul.shape[0]:
            print('error')
        else:
            print(data.shape, allrul.shape)

        plt.plot([i for i in range(len(data))], data[:, displayidx], )
    plt.savefig('train.jpg')
    plt.show()


    # for batteryname in metadata[0] + metadata[1]:
    #     name = 'ne_data/'+batteryname+'v3.npy'
    #     a0 = np.load(name, allow_pickle=True)
    #     seqlen = a0.shape[0]
    #     # a0 = a0.T
    #     if seqlen >= 500:
    #         plt.plot([i for i in range(500)], filter_out_training_extremes(a0[:, 0][:500], threshold=0.1))
    #     # print(a0[0:20, 0])
    # plt.show()

    for batteryname in metadata[2]:
        name = 'ne_data/' + batteryname + 'v3.npy'
        a0 = np.load(name, allow_pickle=True)
        seqlen = a0.shape[0]
        # a0 = a0.T
        plt.plot([i for i in range(seqlen)], a0[:, displayidx])
        # print(a0[0][0:5])

    plt.savefig('test.jpg')
    plt.show()
