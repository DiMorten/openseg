import os
import sys
import time
import numpy as np

from scipy import stats

from skimage import io

from sklearn import metrics

# def fit_weibull(scores):
    
#     # Fitting model.
#     args['thresholds'] = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
#                   0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
#     params = stats.weibull_max.fit(scores - scores.max(), loc=0,  scale=100)
    
#     scr_t_list = []
#     for i, t in enumerate(args['thresholds']):
#         x_threshold = stats.weibull_max.ppf(1.0 - t, *params)
#         scr_t_list.append(x_threshold + scores.max())
    
#     scr_t = np.asarray(scr_t_list)
    
#     return scr_t
    

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



##########################################################################################
##########################################################################################
def main(args):

    epoch = args['epoch_num']

    files = [f for f in os.listdir(args['bas_dir']) if os.path.isfile(os.path.join(args['bas_dir'], f)) and '_tru_' in f]

    tru_list = []
    scr_list = []
    prd_list = []

    tic = time.time()

    for i, f in enumerate(files):

        tru_path = os.path.join(args['bas_dir'], f)
        prd_path = os.path.join(args['img_dir'], f.replace('_tru_', '_prev_'))
        scr_path = os.path.join(args['img_dir'], f.replace('_tru_', '_scor_').replace('.png', '.npy'))

        try:
            tru = io.imread(tru_path)
            prd = io.imread(prd_path)
            scr = np.load(scr_path)
        except:
            print('Error in loading sample "' + f + '"')
            exit(1)

        tru_list.extend(tru.ravel().tolist())
        prd_list.extend(prd.ravel().tolist())
        scr_list.extend(scr.ravel().tolist())
        
    toc = time.time()
    print('Finished Reading Dataset. Elapsed time %.0f' % (toc - tic))

    tru_list = np.asarray(tru_list)
    prd_list = np.asarray(prd_list)
    scr_list = np.asarray(scr_list)

        
    scr_thresholds = np.quantile(scr_list, args['thresholds']).tolist()

    cm_list = []
    acc_known_list = []
    pre_unk_list = []
    rec_unk_list = []
    acc_unknown_list = []
    acc_mean_list = []
    acc_bal_list = []
    kappa_list = []

    for t, scr_t in zip(args['thresholds'], scr_thresholds):
        
        tic = time.time()
        
        prd_list[scr_list < scr_t] = args['n_known']
        
        prd_np = prd_list.ravel()
        tru_np = tru_list.ravel()

        tru_valid = tru_np[tru_np < 5]
        prd_valid = prd_np[tru_np < 5]

        print('Computing CM...')
        cm = metrics.confusion_matrix(tru_valid, prd_valid)

        print('Computing Accs...')
        tru_known = 0.0
        sum_known = 0.0

        for c in range(args['n_known']):
            tru_known += float(cm[c, c])
            sum_known += float(cm[c, :].sum())

        acc_known = float(tru_known) / float(sum_known)

        tru_unknown = float(cm[args['n_known'], args['n_known']])
        sum_unknown_real = float(cm[args['n_known'], :].sum())
        sum_unknown_pred = float(cm[:, args['n_known']].sum())

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
        print('OpenPCA Thresholding %.4f - Acc. Known: %.2f%%, Acc. Unk.: %.2f%%, Pre. Unk.: %.2f%%, Rec. Unk.: %.2f%%, Balanced Acc.: %.2f%%, Kappa: %.2f%%, Time: %.0fs' % (t, acc_known * 100.0, acc_unknown * 100.0, pre_unknown * 100.0, rec_unknown * 100.0, bal * 100.0, kap * 100.0, toc - tic))

        acc_known_list.append(acc_known)
        pre_unk_list.append(pre_unknown)
        rec_unk_list.append(rec_unknown)
        acc_unknown_list.append(acc_unknown)
        acc_mean_list.append(acc_mean)
        acc_bal_list.append(bal)
        kappa_list.append(kap)

    args['thresholds'] = np.asarray(args['thresholds'])
    metric_values={}
    metric_values['acc_known_list'] = np.asarray(acc_known_list) * 100
    metric_values['pre_unk_list'] = np.asarray(pre_unk_list) * 100
    metric_values['rec_unk_list'] = np.asarray(rec_unk_list) * 100
    metric_values['acc_unknown_list'] = np.asarray(acc_unknown_list) * 100
    metric_values['acc_mean_list'] = np.asarray(acc_mean_list) * 100
    metric_values['acc_bal_list'] = np.asarray(acc_bal_list) * 100
    metric_values['kappa_list'] = np.asarray(kappa_list) * 100
    return metric_values
    '''
    with open(args['out_path'], 'w') as file:
        
        for i in range(len(args['thresholds'])):

            file.write('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (scr_thresholds[i],
                                                                    acc_known_list[i],
                                                                    acc_unknown_list[i],
                                                                    pre_unk_list[i],
                                                                    rec_unk_list[i],
                                                                    acc_bal_list[i],
                                                                    kappa_list[i]))
    '''