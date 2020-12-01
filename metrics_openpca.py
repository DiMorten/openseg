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

##########################################################################################
##########################################################################################
##########################################################################################

epoch = 200

conv_name = sys.argv[1]

dataset_name = sys.argv[2]

n_known = 4
class_unk = int(sys.argv[3])

img_dir = './outputs/%s_%s_openpca_%s/epoch_%d/' % (conv_name, dataset_name, str(class_unk), epoch)

# img_dir = './outputs/%s_%s_openpca_full_%s/epoch_%d/' % (conv_name, dataset_name, str(class_unk), epoch)

out_path = './metrics/%s_%s_%s_openpca.txt' % (conv_name, dataset_name, str(class_unk))

##########################################################################################
##########################################################################################

thresholds = []
if conv_name == 'unet':
    if dataset_name == 'Vaihingen':
        thresholds = [-320, -280, -240, -200, -180, -160, -140, -120, -100, -80, -60]
    elif dataset_name == 'Potsdam':
         thresholds = [-40, 0, 40, 80, 120, 160, 200, 220, 240, 260, 280, 300]

elif conv_name == 'fcnresnet50':

    # FCN ResNet-50.
    thresholds = [-1200, -1100, -1000, -900, -800, -700, -600, -550, -500, -450, -400, -350, -300]
    
elif conv_name == 'fcnresnext50':

    # FCN ResNext-50.
    thresholds = [-1200, -1100, -1000, -900, -800, -700, -600, -550, -500, -450, -400, -350, -300]
    
elif conv_name == 'fcnwideresnet50':

    # FCN Wide ResNet-50.
    thresholds = [-1200, -1100, -1000, -900, -800, -700, -600, -550, -500, -450, -400, -350, -300]
    
elif conv_name == 'fcndensenet121':

    # FCN DenseNet-121.
    thresholds = [-280, -260, -240, -220, -200, -180, -160, -140, -120, -100, -80]
    
elif conv_name == 'fcndensenet121pretrained':

    # FCN DenseNet-121.
    thresholds = [-280, -260, -240, -220, -200, -180, -160, -140, -120, -100, -80]
    
elif conv_name == 'fcnvgg19':

    # FCN VGG-19 BN.
    thresholds = [-280, -260, -240, -220, -200, -180, -160, -140, -120, -100, -80]
    
elif conv_name == 'fcnvgg19pretrained':

    # FCN VGG-19 BN.
    thresholds = [-280, -260, -240, -220, -200, -180, -160, -140, -120, -100, -80]
    
elif conv_name == 'fcninceptionv3':

    # FCN Inception v3.
    thresholds = [-280, -260, -240, -220, -200, -180, -160, -140, -120, -100, -80]
    
elif conv_name == 'fcnmobilenetv2':

    # FCN MobileNet v2.
    thresholds = [-800, -760, -720, -700, -680, -660, -640, -620, -600]
    
elif conv_name == 'segnet':

    # SegNet.
    thresholds = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120, 140]

files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and '_true_' in f]

tru_list = []
pro_list = []
prd_list = []

tic = time.time()

for i, f in enumerate(files):

    tru_path = os.path.join(img_dir, f)
    prd_path = os.path.join(img_dir, f.replace('_true_', '_prev_'))
    pro_path = os.path.join(img_dir, f.replace('_true_', '_scor_').replace('.png', '.npy'))

    try:
        tru = io.imread(tru_path)
        prd = io.imread(prd_path)
        pro = np.load(pro_path)
    except:
        print('Error in loading sample "' + f + '"')
        exit(1)

    tru_list.append(tru)
    prd_list.append(prd)
    pro_list.append(pro)
    
toc = time.time()
print('Finished Reading Dataset. Elapsed time %.0f' % (toc - tic))

tru_list = np.asarray(tru_list)
pro_list = np.asarray(pro_list)
prd_list = np.asarray(prd_list)

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
    
    prd_list[pro_list < t] = n_known
    
    prd_np = prd_list.ravel()
    tru_np = tru_list.ravel()

    tru_valid = tru_np[tru_np < 5]
    prd_valid = prd_np[tru_np < 5]

    print('Computing CM...')
    cm = metrics.confusion_matrix(tru_valid, prd_valid)

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
    bal = metrics.balanced_accuracy_score(tru_valid, prd_valid)
    
    print('Computing Kappa...')
    kap = metrics.cohen_kappa_score(tru_valid, prd_valid)

    toc = time.time()
    print('OpenPCA Thresholding %.2f - Acc. Known: %.2f%%, Acc. Unk.: %.2f%%, Pre. Unk.: %.2f%%, Rec. Unk.: %.2f%%, Balanced Acc.: %.2f%%, Kappa: %.2f%%, Time: %.0fs' % (t, acc_known * 100.0, acc_unknown * 100.0, pre_unknown * 100.0, rec_unknown * 100.0, bal * 100.0, kap * 100.0, toc - tic))

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

with open(out_path, 'w') as file:
    
    for i in range(len(thresholds)):

        file.write('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (acc_known_list[i],
                                                             acc_unknown_list[i],
                                                             pre_unk_list[i],
                                                             rec_unk_list[i],
                                                             acc_bal_list[i],
                                                             kappa_list[i]))
