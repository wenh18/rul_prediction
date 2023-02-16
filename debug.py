import pickle
import matplotlib.pyplot as plt
# new_test = ['9-6', '4-5', '1-2', '10-7', '1-1', '6-1', '6-6', '9-4', '10-4', '8-5', '5-3', '10-6',
#             '2-5', '6-2', '3-1', '8-8', '8-1', '8-6', '7-6', '6-8', '7-5', '10-1']
# # name = 'our_data/9-6'
# prefix = 209
# for ba in new_test:
#     name = 'our_data/'+ba
#     with open(name + '.pkl', 'rb') as f:
#         a = pickle.load(f)
#     # print(a['9-6']['dq'])
#     # import pdb;pdb.set_trace()
#     caps = []
#     for k,v in a[ba]['dq'].items():
#         caps.append(v)
#     plt.plot([i for i in range(len(caps[9:prefix]))], caps[9:prefix])
#     plt.show()

# import numpy as np
# a = np.load('our_data/1-1v2.npy')
# arul = np.load('our_data/1-1_rulv2.npy')
# b = np.load('our_data/1-1.npy')
# brul = np.load('our_data/1-1_rul.npy')
# print(a.shape, arul.shape, b.shape, brul.shape)

# a = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [3, 4], [4, 3]]#, [3, 5], [4, 3], [4, 5], [5, 4], [5, 3]]
# ratio = []
# for i in a:
#     ratio.append(i[0]/i[1])
# ratio.sort()
# print(ratio)
# print(len(a))

a =  [0.0717, -0.0068, -0.0197, -0.2726, -0.0043,  0.1435, -0.0226,  0.0238,
        -0.1421,  0.0297,  0.0549,  0.0793, -0.0738,  0.1241, -0.0446,  0.0019,
        -0.0375, -0.0781,  0.0224,  0.1061,  0.0394,  0.0119, -0.1692,  0.1793,
        -0.0069,  0.0902, -0.0416,  0.0162, -0.0237,  0.1104,  0.0093,  0.0753]
b = [ 0.0131,  0.1271,  0.0291,  0.0726,  0.1552, -0.0996, -0.0153,  0.1304,
         -0.0871,  0.1261,  0.1100,  0.1080,  0.0304, -0.1057, -0.1697,  0.0828,
          0.0962, -0.1761,  0.0651, -0.0312, -0.0410,  0.1511,  0.0180,  0.0849,
         -0.1729, -0.1180,  0.1351, -0.1546, -0.0051, -0.1007, -0.0551, -0.0967]
result = 0
for i in range(len(a)):
    result += a[i] * b[i]

print(result+0.0316)