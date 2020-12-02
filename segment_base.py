import datetime
import os
import random
import time
import gc
import sys
import numpy as np

import scipy.spatial.distance as spd

from skimage import io
from skimage import util

from sklearn import metrics

from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F

import list_dataset
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d, LovaszLoss, FocalLoss2d

from openmax import *

cudnn.benchmark = True

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


# Main function.
def main(args):


	hidden = []
	if '_' in args['hidden_classes']:
		hidden = [int(h) for h in args['hidden_classes'].split('_')]
	else:
		hidden = [int(args['hidden_classes'])]

	num_known_classes = list_dataset.num_classes - len(hidden)
	num_unknown_classes = len(hidden)

	weights = []
	if 4 not in hidden:
		weights = [1.0 for i in range(num_known_classes)]
		weights[-1] = 2.0
	else:
		weights = [1.0 for i in range(num_known_classes)]

	weights = torch.FloatTensor(weights)

	# Setting experiment name.
	args['exp_name'] = args['conv_name'] + '_' + args['dataset_name'] + '_base_' + args['hidden_classes']

	# Setting device [0|1|2].
	args['device'] = 0


	# Setting network architecture.
	if (args['conv_name'] == 'unet'):

		net = UNet(3, num_classes=list_dataset.num_classes, hidden_classes=hidden).cuda(args['device'])
		
	elif (args['conv_name'] == 'fcnresnet50'):

		net = FCNResNet50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
		
	elif (args['conv_name'] == 'fcnresnet50pretrained'):

		net = FCNResNet50(3, num_classes=list_dataset.num_classes, pretrained=True, skip=True, hidden_classes=hidden).cuda(args['device'])
		args['lr'] *= 0.1
		
	elif (args['conv_name'] == 'fcnresnext50'):

		net = FCNResNext50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
		
	elif (args['conv_name'] == 'fcnwideresnet50'):

		net = FCNWideResNet50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
		
	elif (args['conv_name'] == 'fcndensenet121'):

		net = FCNDenseNet121(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
		
	elif (args['conv_name'] == 'fcndensenet121pretrained'):

		net = FCNDenseNet121(3, num_classes=list_dataset.num_classes, pretrained=True, skip=True, hidden_classes=hidden).cuda(args['device'])
		args['lr'] *= 0.1
		
	elif (args['conv_name'] == 'fcnvgg19'):

		net = FCNVGG19(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
		
	elif (args['conv_name'] == 'fcnvgg19pretrained'):

		net = FCNVGG19(3, num_classes=list_dataset.num_classes, pretrained=True, skip=True, hidden_classes=hidden).cuda(args['device'])
		args['lr'] *= 0.1
		
	elif (args['conv_name'] == 'fcninceptionv3'):

		net = FCNInceptionv3(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])

	elif (args['conv_name'] == 'segnet'):

		net = SegNet(3, num_classes=list_dataset.num_classes, hidden_classes=hidden).cuda(args['device'])
		
	print(net)
	
	curr_epoch = 1
	args['best_record'] = {'epoch': 0, 'lr': 1e-4, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'iou': 0}

	# Setting datasets.
	train_set = list_dataset.ListDataset(args['dataset_name'], 'Train', (args['h_size'], args['w_size']), 'statistical', hidden)
	train_loader = DataLoader(train_set, batch_size=1, num_workers=args['num_workers'], shuffle=True)

	test_set = list_dataset.ListDataset(args['dataset_name'], 'Test', (args['h_size'], args['w_size']), 'statistical', hidden)
	test_loader = DataLoader(test_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)

	# Setting criterion.
	criterion = CrossEntropyLoss2d(weight=weights, size_average=False, ignore_index=5).cuda(args['device'])
	if args['baseline_train']==True:
		# Setting optimizer.
		optimizer = optim.Adam([
			{'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
			'lr': 2 * args['lr']},
			{'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
			'lr': args['lr'], 'weight_decay': args['weight_decay']}
		], betas=(args['momentum'], 0.99))

		scheduler = optim.lr_scheduler.StepLR(optimizer, args['epoch_num'] // 3, 0.2)

		# Making sure checkpoint and output directories are created.
		check_mkdir(args['ckpt_path'])
		check_mkdir(os.path.join(args['ckpt_path'], args['exp_name']))
		check_mkdir(args['outp_path'])
		check_mkdir(os.path.join(args['outp_path'], args['exp_name']))

		# Writing training args to experiment log file.
		open(os.path.join(args['ckpt_path'], args['exp_name'], str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')
		
		# Iterating over epochs.
		for epoch in range(curr_epoch, args['epoch_num'] + 1):

			# Training function.
			train(train_loader, net, criterion, optimizer, epoch, num_known_classes, num_unknown_classes, hidden, args)

			if epoch % args['test_freq'] == 0:
				
				torch.save(net.state_dict(), os.path.join(args['ckpt_path'], args['exp_name'], 'model_' + str(epoch) + '.pth'))
				torch.save(optimizer.state_dict(), os.path.join(args['ckpt_path'], args['exp_name'], 'opt_' + str(epoch) + '.pth'))
				
				# Computing test.
				test(test_loader, net, criterion, optimizer, epoch, num_known_classes, num_unknown_classes, hidden, args, True, True) #epoch % args['save_freq'] == 0)
			
			scheduler.step()
			
		print('Exiting training...')
	else:
		#args['exp_name'] = conv_name + '_' + dataset_name + '_base_' + args['hidden_classes']
		pretrained_path = os.path.join(args['ckpt_path'], args['exp_name'], 'model_' + str(args['epoch_num']) + '.pth')
		print('Loading pretrained weights from file "' + pretrained_path + '"')
		net.load_state_dict(torch.load(pretrained_path))        
	
	print("Testing baseline...")
	import warnings
	warnings.filterwarnings("ignore")
	if args['baseline_train']==True:
		test(test_loader, net, criterion, optimizer, epoch, num_known_classes, num_unknown_classes, hidden, args, True, True) #epoch % args['save_freq'] == 0)
	else:
		test(test_loader, net, criterion, [], args['epoch_num'], num_known_classes, num_unknown_classes, hidden, args, True, True) #epoch % args['save_freq'] == 0)
# Training function.
def train(train_loader, net, criterion, optimizer, epoch, num_known_classes, num_unknown_classes, hidden, args):

	# Setting network for training mode.
	net.train()

	# Average Meter for batch loss.
	train_loss = list()

	# Iterating over batches.
	for i, data in enumerate(train_loader):

		# Obtaining images, labels and paths for batch.
		inps, labs, true, img_name = data

		# Casting tensors to cuda.
		inps, labs, true = inps.cuda(args['device']), labs.cuda(args['device']), true.cuda(args['device'])

		inps.squeeze_(0)
		labs.squeeze_(0)
		true.squeeze_(0)

		# Casting to cuda variables.
		inps = Variable(inps).cuda(args['device'])
		labs = Variable(labs).cuda(args['device'])
		true = Variable(true).cuda(args['device'])

		# Clears the gradients of optimizer.
		optimizer.zero_grad()

		# Forwarding.
		outs = net(inps)
		soft_outs = F.softmax(outs, dim=1)

		# Obtaining predictions.
		prds = soft_outs.data.max(1)[1]

		# Computing loss.
		loss = criterion(outs, labs)
			
		# Computing backpropagation.
		loss.backward()
		optimizer.step()
		
		# Appending images for epoch loss calculation.
		prds = prds.squeeze_(1).squeeze_(0).cpu().numpy()
		
		inps_np = inps.detach().squeeze(0).cpu().numpy()
		labs_np = labs.detach().squeeze(0).cpu().numpy()
		true_np = true.detach().squeeze(0).cpu().numpy()

		# Updating loss meter.
		train_loss.append(loss.data.item())

		# Printing.
		if (i + 1) % args['print_freq'] == 0:
			print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (epoch, i + 1, len(train_loader), np.asarray(train_loss).mean()))
	
	sys.stdout.flush()

def test(test_loader, net, criterion, optimizer, epoch, num_known_classes, num_unknown_classes, hidden, args, save_images, save_model):

	# Setting network for evaluation mode.
	net.eval()
	
	with torch.no_grad():

		# Creating output directory.
		check_mkdir(os.path.join(args['outp_path'], args['exp_name'], 'epoch_' + str(epoch)))

		# Iterating over batches.
		for i, data in enumerate(test_loader):
			
			print('Test Batch %d/%d' % (i + 1, len(test_loader)))

			# Obtaining images, labels and paths for batch.
			inps_batch, labs_batch, true_batch, img_name = data

			inps_batch = inps_batch.squeeze()
			labs_batch = labs_batch.squeeze()
			true_batch = true_batch.squeeze()

			# Iterating over patches inside batch.
			for j in range(inps_batch.size(0)):
			
				print('    Test MiniBatch %d/%d' % (j + 1, inps_batch.size(0)))
				sys.stdout.flush()
				
				tic = time.time()
				
				for k in range(inps_batch.size(1)):
					
					inps = inps_batch[j, k].unsqueeze(0)
					labs = labs_batch[j, k].unsqueeze(0)
					true = true_batch[j, k].unsqueeze(0)

					# Casting tensors to cuda.
					inps, labs, true = inps.cuda(args['device']), labs.cuda(args['device']), true.cuda(args['device'])
					
					# Casting to cuda variables.
					inps = Variable(inps).cuda(args['device'])
					labs = Variable(labs).cuda(args['device'])
					true = Variable(true).cuda(args['device'])
					
					# Forwarding.
					if args['conv_name'] == 'unet':
						outs, dec1, dec2, dec3, dec4 = net(inps, feat=True)
					elif args['conv_name'] == 'fcnresnet50':
						outs, classif1, fv2 = net(inps, feat=True)
					elif args['conv_name'] == 'fcnresnext50':
						outs, classif1, fv2 = net(inps, feat=True)
					elif args['conv_name'] == 'fcnwideresnet50':
						outs, classif1, fv2 = net(inps, feat=True)
					elif args['conv_name'] == 'fcndensenet121':
						outs, classif1, fv2 = net(inps, feat=True)
					elif args['conv_name'] == 'fcndensenet121pretrained':
						outs, classif1, fv2 = net(inps, feat=True)
					elif args['conv_name'] == 'fcnvgg19':
						outs, classif1, fv3 = net(inps, feat=True)
					elif args['conv_name'] == 'fcnvgg19pretrained':
						outs, classif1, fv3 = net(inps, feat=True)
					elif args['conv_name'] == 'fcninceptionv3':
						outs, classif1, fv4 = net(inps, feat=True)
					elif args['conv_name'] == 'fcnmobilenetv2':
						outs, classif1, fv3 = net(inps, feat=True)
					elif args['conv_name'] == 'segnet':
						outs, x_10d, x_20d = net(inps, feat=True)
					
					# Computing probabilities.
					soft_outs = F.softmax(outs, dim=1)
					
					# Obtaining prior predictions.
					prds = soft_outs.data.max(1)[1]
					
					# Obtaining posterior predictions.
					if args['conv_name'] == 'unet':
						feat_flat = torch.cat([outs, dec1, dec2, dec3], 1)
					elif args['conv_name'] == 'fcnresnet50':
						feat_flat = torch.cat([outs, classif1, fv2], 1)
					elif args['conv_name'] == 'fcnresnext50':
						feat_flat = torch.cat([outs, classif1, fv2], 1)
					elif args['conv_name'] == 'fcnwideresnet50':
						feat_flat = torch.cat([outs, classif1, fv2], 1)
					elif args['conv_name'] == 'fcndensenet121':
						feat_flat = torch.cat([outs, classif1, fv2], 1)
					elif args['conv_name'] == 'fcndensenet121pretrained':
						feat_flat = torch.cat([outs, classif1, fv2], 1)
					elif args['conv_name'] == 'fcnvgg19':
						feat_flat = torch.cat([outs, classif1, fv3], 1)
					elif args['conv_name'] == 'fcnvgg19pretrained':
						feat_flat = torch.cat([outs, classif1, fv3], 1)
					elif args['conv_name'] == 'fcninceptionv3':
						feat_flat = torch.cat([outs, classif1, fv4], 1)
					elif args['conv_name'] == 'fcnmobilenetv2':
						feat_flat = torch.cat([outs, classif1, fv3], 1)
					elif args['conv_name'] == 'segnet':
						feat_flat = torch.cat([outs, x_10d, x_20d], 1)
					feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(feat_flat.size(0) * feat_flat.size(2) * feat_flat.size(3), feat_flat.size(1)).cpu().numpy()
					prds_flat = prds.cpu().numpy().ravel()
					true_flat = true.cpu().numpy().ravel()
					
					# Appending images for epoch loss calculation.
					inps_np = inps.detach().squeeze(0).cpu().numpy()
					labs_np = labs.detach().squeeze(0).cpu().numpy()
					true_np = true.detach().squeeze(0).cpu().numpy()
					
					print("Test patch result shape",labs_np.shape,true_np.shape)
					accuracy_value = metrics.accuracy_score(true_np,labs_np)
					print("Accuracy value",accuracy_value)                    
					
					# Saving predictions.
					if (save_images):
						
						imag_path = os.path.join(args['outp_path'], args['exp_name'], 'epoch_' + str(epoch), img_name[0].replace('.tif', '_img_' + str(j) + '_' + str(k) + '.png'))
						mask_path = os.path.join(args['outp_path'], args['exp_name'], 'epoch_' + str(epoch), img_name[0].replace('.tif', '_msk_' + str(j) + '_' + str(k) + '.png'))
						true_path = os.path.join(args['outp_path'], args['exp_name'], 'epoch_' + str(epoch), img_name[0].replace('.tif', '_tru_' + str(j) + '_' + str(k) + '.png'))
						pred_path = os.path.join(args['outp_path'], args['exp_name'], 'epoch_' + str(epoch), img_name[0].replace('.tif', '_prd_' + str(j) + '_' + str(k) + '.png'))
						
						io.imsave(imag_path, np.transpose(inps_np, (1, 2, 0)))
						io.imsave(mask_path, util.img_as_ubyte(labs_np))
						io.imsave(true_path, util.img_as_ubyte(true_np))
						io.imsave(pred_path, util.img_as_ubyte(prds.cpu().squeeze().numpy()))
				
				toc = time.time()
				print('        Elapsed Time: %.2f' % (toc - tic))
				
		sys.stdout.flush()
		
		if save_model:

			torch.save(net.state_dict(), os.path.join(args['ckpt_path'], args['exp_name'], 'model_' + str(epoch) + '.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args['ckpt_path'], args['exp_name'], 'opt_' + str(epoch) + '.pth'))

if __name__ == '__main__':
	main(args)
