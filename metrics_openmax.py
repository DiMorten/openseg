import os
import sys
import time
import numpy as np

from skimage import io

from sklearn import metrics

'''
Classes:
    0 = Street
    1 = Building
    2 = Grass
    3 = Tree
    4 = Car
    5 = Surfaces
    6 = Boundaries
'''

##################################################################################
##################################################################################
##################################################################################

epoch = 1200

conv_name = 'fcnwideresnet50'

dataset_name = 'Vaihingen'

if dataset_name == 'Potsdam':
    epoch = 600

n_known = 4
class_unk = int(1)

bas_dir = './outputs/%s_%s_base_%s/epoch_%d/' % (conv_name, dataset_name, str(class_unk), epoch)
img_dir = './outputs/%s_%s_openmax_%s/epoch_%d/' % (conv_name, dataset_name, str(class_unk), epoch)

out_path = './metrics/%s_%s_%s_openmax.txt' % (conv_name, dataset_name, str(class_unk))

##################################################################################
##################################################################################

thresholds = []
if dataset_name == 'Vaihingen':
#     if os.path.isfile(out_path):
#         thresholds = [0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995]
#     else:
#         thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
#                       0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                  0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995]
elif dataset_name == 'Potsdam':
    thresholds = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

files = [f for f in os.listdir(bas_dir) if os.path.isfile(os.path.join(bas_dir, f)) and '_tru_' in f]

tru_list = []
pro_list = []

tic = time.time()

for i, f in enumerate(files):

    tru_path = os.path.join(bas_dir, f)
    pro_path = os.path.join(img_dir, f.replace('_tru_', '_prob_').replace('.png', '.npy'))

    try:
        tru = io.imread(tru_path)
        pro = np.load(pro_path)
    except:
        print('Error in loading sample "' + f + '"')
        break

    tru_list.append(tru)
    pro_list.append(pro)
    
toc = time.time()
print('Finished Reading Dataset. Elapsed time %.0f' % (toc - tic))

pro_list = np.asarray(pro_list)
tru_list = np.asarray(tru_list)

cm_list = []
acc_known_list = []
pre_unk_list = []
rec_unk_list = []
acc_unknown_list = []
acc_mean_list = []
acc_bal_list = []
kappa_list = []

for t in thresholds:
    
    tic = time.time()
    pos_np = pro_list.argmax(axis=1)
    pos_np[pro_list.max(axis=1) < t] = n_known
    
    pos_np = pos_np.ravel()
    tru_np = tru_list.ravel()

    tru_valid = tru_np[tru_np < 5]
    pos_valid = pos_np[tru_np < 5]

    print('Computing CM...')
    cm = metrics.confusion_matrix(tru_valid, pos_valid)

    print('Computing Accs...')
    tru_known = 0.0
    sum_known = 0.0

    for c in range(n_known):
        tru_known += float(cm[c, c])
        sum_known += float(cm[c, :].sum())

    acc_known = float(tru_known) / float(sum_known)

    tru_unknown = float(cm[n_known, n_known])
    sum_unknown_real = float(cm[n_known, :].sum())
    sum_unknown_pred = float(cm[:, n_known].sum())

    pre_unknown = 0.0
    rec_unknown = 0.0
    
    if sum_unknown_pred != 0.0:
        pre_unknown = float(tru_unknown) / float(sum_unknown_pred)
    if sum_unknown_real != 0.0:
        rec_unknown = float(tru_unknown) / float(sum_unknown_real)
        
    acc_unknown = (tru_known + tru_unknown) / (sum_known + sum_unknown_real)

    acc_mean = (acc_known + acc_unknown) / 2.0

    print('Computing Balanced Acc...')
    bal = metrics.balanced_accuracy_score(tru_valid, pos_valid)
    
    print('Computing Kappa...')
    kap = metrics.cohen_kappa_score(tru_valid, pos_valid)

    toc = time.time()
    print('OpenMax Thresholding %.3f - Acc. Known: %.2f%%, Acc. Unk.: %.2f%%, Pre. Unk.: %.2f%%, Rec. Unk.: %.2f%%, Balanced Acc.: %.2f%%, Kappa: %.2f%%, Time: %.0fs' % (t, acc_known * 100.0, acc_unknown * 100.0, pre_unknown * 100.0, rec_unknown * 100.0, bal * 100.0, kap * 100.0, toc - tic))

    acc_known_list.append(acc_known)
    pre_unk_list.append(pre_unknown)
    rec_unk_list.append(rec_unknown)
    acc_unknown_list.append(acc_unknown)
    acc_mean_list.append(acc_mean)
    acc_bal_list.append(bal)
    kappa_list.append(kap)

thresholds = np.asarray(thresholds)
acc_known_list = np.asarray(acc_known_list) * 100
pre_unk_list = np.asarray(pre_unk_list) * 100
rec_unk_list = np.asarray(rec_unk_list) * 100
acc_unknown_list = np.asarray(acc_unknown_list) * 100
acc_mean_list = np.asarray(acc_mean_list) * 100
acc_bal_list = np.asarray(acc_bal_list) * 100
kappa_list = np.asarray(kappa_list) * 100

# if os.path.isfile(out_path):
#     file_mode = 'a'
# else:
#     file_mode = 'w'
file_mode = 'w'

with open(out_path, file_mode) as file:
    
    for i in range(len(thresholds)):

        file.write('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (acc_known_list[i],
                                                             acc_unknown_list[i],
                                                             pre_unk_list[i],
                                                             rec_unk_list[i],
                                                             acc_bal_list[i],
                                                             kappa_list[i]))
