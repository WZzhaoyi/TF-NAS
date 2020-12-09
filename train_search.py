import os
import sys
import time
import glob
import logging
import argparse
import pickle
import copy
import numpy as np
import warnings
import datetime
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from tools.utils import AverageMeter, accuracy, SegmentationMetric
from tools.utils import count_parameters_in_MB
from tools.utils import create_exp_dir
from tools.config import mc_mask_dddict, lat_lookup_key_dddict
from models.seg_model_search import Network
from parsing_model import get_op_and_depth_weights
from parsing_model import parse_architecture
from parsing_model import get_mc_num_dddict
from dataset import get_segmentation_dataset
from dataset import make_data_sampler
from dataset import make_batch_data_sampler
from loss import MixSoftmaxCrossEntropyLoss
from loss import reduce_loss_dict

parser = argparse.ArgumentParser("searching TF-NAS")
# various path
parser.add_argument('--img_root', type=str, help='image root path (ImageNet train set)')
parser.add_argument('--train_list', type=str, default="./dataset/ImageNet-100-effb0_train_cls_ratio0.8.txt",
					help='training image list')
parser.add_argument('--val_list', type=str, default="./dataset/ImageNet-100-effb0_val_cls_ratio0.8.txt",
					help='validating image list')
parser.add_argument('--lookup_path', type=str, default="./latency_pkl/latency_gpu.pkl",
					help='path of lookup table')
parser.add_argument('--save', type=str, default='./checkpoints', help='model and log saving path')

# training hyper-parameters
parser.add_argument('--print_freq', type=float, default=100, help='print frequency')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--epochs', type=int, default=50, help='num of total training epochs')
parser.add_argument('--search_epoch', type=int, default=10, help='start epoch for search')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--w_lr', type=float, default=0.025, help='learning rate for weights')
parser.add_argument('--w_mom', type=float, default=0.9, help='momentum for weights')
parser.add_argument('--w_wd', type=float, default=1e-5, help='weight decay for weights')
parser.add_argument('--a_lr', type=float, default=0.01, help='learning rate for arch')
parser.add_argument('--a_wd', type=float, default=5e-4, help='weight decay for arch')
parser.add_argument('--a_beta1', type=float, default=0.5, help='beta1 for arch')
parser.add_argument('--a_beta2', type=float, default=0.999, help='beta2 for arch')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--T', type=float, default=5.0, help='temperature for gumbel softmax')
parser.add_argument('--T_decay', type=float, default=0.96, help='temperature decay')
parser.add_argument('--num_classes', type=int, default=19, help='class number of training set')
parser.add_argument('--aux', type=bool, default=False, help='aux train for w')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--distributed', type=int, default=0, help='distributed')
parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
parser.add_argument('--rank', default=0, help='rank of current process')
parser.add_argument('--word_size', default=1, type=int, help="word size")
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456', help="init-method")

# hyper parameters
parser.add_argument('--lambda_lat', type=float, default=0.1, help='trade off for latency')
parser.add_argument('--target_lat', type=float, default=15.0, help='the target latency')


args = parser.parse_args()

args.save = os.path.join(args.save, 'search-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(logdir=os.path.join(args.save, 'log'))

def main():
	if not torch.cuda.is_available():
		logging.info('No GPU device available')
		sys.exit(1)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	cudnn.enabled=True
	cudnn.benchmark = True
	logging.info("args = %s", args)

	with open(args.lookup_path, 'rb') as f:
		lat_lookup = pickle.load(f)

	mc_maxnum_dddict = get_mc_num_dddict(mc_mask_dddict, is_max=True)
	if args.distributed or args.num_gpus > 1:
		dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)
	model = Network(args.num_classes, mc_maxnum_dddict, lat_lookup, aux=args.aux)
	model = torch.nn.DataParallel(model).cuda()
	logging.info("param size = %fMB", count_parameters_in_MB(model))

	# save initial model
	model_path = os.path.join(args.save, 'searched_model_00.pth.tar')
	torch.save({
			'state_dict': model.state_dict(),
			'mc_mask_dddict': mc_mask_dddict,
		}, model_path)

	# get lr list
	lr_list = []
	optimizer_w = torch.optim.SGD(
					model.module.weight_parameters(),
					lr = args.w_lr,
					momentum = args.w_mom,
					weight_decay = args.w_wd)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_w, float(args.epochs))
	for _ in range(args.epochs):
		lr = scheduler.get_lr()[0]
		lr_list.append(lr)
		scheduler.step()
	del model
	del optimizer_w
	del scheduler

	criterion = MixSoftmaxCrossEntropyLoss(aux=args.aux)
	criterion = criterion.cuda()

	input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
	data_kwargs = {'transform': input_transform, 'base_size': 1024, 'crop_size': (512, 1024)}
	train_dataset = get_segmentation_dataset(split='train', mode='train', **data_kwargs)
	val_dataset = get_segmentation_dataset(split='val', mode='val', **data_kwargs)

	# iters_per_epoch = len(train_dataset) // args.batch_size
	# max_iters = args.epochs * iters_per_epoch

	# train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
	# train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, iters_per_epoch, drop_last=True)
	# val_sampler = make_data_sampler(val_dataset, shuffle=True, distributed=args.distributed)
	# val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size, drop_last=True)
	
	train_queue = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)

	val_queue = torch.utils.data.DataLoader(
		dataset=val_dataset,
		batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)

	for epoch in range(args.epochs):
		mc_num_dddict = get_mc_num_dddict(mc_mask_dddict)
		model = Network(args.num_classes, mc_num_dddict, lat_lookup, aux=args.aux)
		model = torch.nn.DataParallel(model).cuda()
		model.module.set_temperature(args.T)

		# load model
		model_path = os.path.join(args.save, 'searched_model_{:02}.pth.tar'.format(epoch))
		state_dict = torch.load(model_path)['state_dict']
		for key in state_dict:
			if 'm_ops' not in key:
				# print('model.{}.data'.format(key))
				exec('model.{}.data = state_dict[key].data'.format(key))
		for stage in mc_mask_dddict:
			for block in mc_mask_dddict[stage]:
				for op_idx in mc_mask_dddict[stage][block]:
					index = torch.nonzero(mc_mask_dddict[stage][block][op_idx]).view(-1)
					index = index.cuda()
					iw = 'model.module.{}.{}.m_ops[{}].inverted_bottleneck.conv.weight.data'.format(stage, block, op_idx)
					iw_key = 'module.{}.{}.m_ops.{}.inverted_bottleneck.conv.weight'.format(stage, block, op_idx)
					exec(iw + ' = torch.index_select(state_dict[iw_key], 0, index).data')
					dw = 'model.module.{}.{}.m_ops[{}].depth_conv.conv.weight.data'.format(stage, block, op_idx)
					dw_key = 'module.{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
					exec(dw + ' = torch.index_select(state_dict[dw_key], 0, index).data')
					pw = 'model.module.{}.{}.m_ops[{}].point_linear.conv.weight.data'.format(stage, block, op_idx)
					pw_key = 'module.{}.{}.m_ops.{}.point_linear.conv.weight'.format(stage, block, op_idx)
					exec(pw + ' = torch.index_select(state_dict[pw_key], 1, index).data')
					if op_idx >= 4:
						se_cr_w = 'model.module.{}.{}.m_ops[{}].squeeze_excite.conv_reduce.weight.data'.format(stage, block, op_idx)
						se_cr_w_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_reduce.weight'.format(stage, block, op_idx)
						exec(se_cr_w + ' = torch.index_select(state_dict[se_cr_w_key], 1, index).data')
						se_cr_b = 'model.module.{}.{}.m_ops[{}].squeeze_excite.conv_reduce.bias.data'.format(stage, block, op_idx)
						se_cr_b_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_reduce.bias'.format(stage, block, op_idx)
						exec(se_cr_b + ' = state_dict[se_cr_b_key].data')
						se_ce_w = 'model.module.{}.{}.m_ops[{}].squeeze_excite.conv_expand.weight.data'.format(stage, block, op_idx)
						se_ce_w_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_expand.weight'.format(stage, block, op_idx)
						exec(se_ce_w + ' = torch.index_select(state_dict[se_ce_w_key], 0, index).data')
						se_ce_b = 'model.module.{}.{}.m_ops[{}].squeeze_excite.conv_expand.bias.data'.format(stage, block, op_idx)
						se_ce_b_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_expand.bias'.format(stage, block, op_idx)
						exec(se_ce_b + ' = torch.index_select(state_dict[se_ce_b_key], 0, index).data')
		del index

		lr = lr_list[epoch]
		optimizer_w = torch.optim.SGD(
						model.module.weight_parameters(),
						lr = lr,
						momentum = args.w_mom,
						weight_decay = args.w_wd)
		optimizer_a = torch.optim.Adam(
						model.module.arch_parameters(),
						lr = args.a_lr,
						betas = (args.a_beta1, args.a_beta2),
						weight_decay = args.a_wd)
		logging.info('Epoch: %d lr: %e T: %e', epoch, lr, args.T)

		# training
		epoch_start = time.time()
		if epoch < args.search_epoch:
			train_acc = train_wo_arch(train_queue, model, criterion, optimizer_w)
			writer.add_scalar('Train/objs_w', train_acc, epoch)
		else:
			train_acc = train_w_arch(train_queue, val_queue, model, criterion, optimizer_w, optimizer_a)
			args.T *= args.T_decay
			writer.add_scalar('Train/objs_w', train_acc[0], epoch)
			writer.add_scalar('Train/objs_a', train_acc[1], epoch)
			train_acc = train_acc[0]
		# logging arch parameters
		logging.info('The current arch parameters are:')
		for param in model.module.log_alphas_parameters():
			param = np.exp(param.detach().cpu().numpy())
			logging.info(' '.join(['{:.6f}'.format(p) for p in param]))
		for param in model.module.betas_parameters():
			param = F.softmax(param.detach().cpu(), dim=-1)
			param = param.numpy()
			logging.info(' '.join(['{:.6f}'.format(p) for p in param]))
		logging.info('Train_acc %f', train_acc)
		epoch_duration = time.time() - epoch_start
		eta_string = str(datetime.timedelta(seconds=int(epoch_duration*(args.epochs-epoch))))
		logging.info('Epoch time: %ds, Estimated Time: %s', epoch_duration, eta_string)
		

		# validation for last 5 epochs
		if args.epochs - epoch < 10 or epoch >= args.search_epoch:
			val_acc = validate(val_queue, model, criterion, epoch, args)
			logging.info('Val_acc pixAcc: %f mIoU: %f', val_acc[0], val_acc[1])
			writer.add_scalar('Val/pixAcc', val_acc[0], epoch)
			writer.add_scalar('Val/mIoU', val_acc[1], epoch)

		# update state_dict
		state_dict_from_model = model.state_dict()
		for key in state_dict:
			if 'm_ops' not in key:
				state_dict[key].data = state_dict_from_model[key].data
		for stage in mc_mask_dddict:
			for block in mc_mask_dddict[stage]:
				for op_idx in mc_mask_dddict[stage][block]:
					index = torch.nonzero(mc_mask_dddict[stage][block][op_idx]).view(-1)
					index = index.cuda()
					iw_key = 'module.{}.{}.m_ops.{}.inverted_bottleneck.conv.weight'.format(stage, block, op_idx)
					state_dict[iw_key].data[index,:,:,:] = state_dict_from_model[iw_key]
					dw_key = 'module.{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
					state_dict[dw_key].data[index,:,:,:] = state_dict_from_model[dw_key]
					pw_key = 'module.{}.{}.m_ops.{}.point_linear.conv.weight'.format(stage, block, op_idx)
					state_dict[pw_key].data[:,index,:,:] = state_dict_from_model[pw_key]
					if op_idx >= 4:
						se_cr_w_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_reduce.weight'.format(stage, block, op_idx)
						state_dict[se_cr_w_key].data[:,index,:,:] = state_dict_from_model[se_cr_w_key]
						se_cr_b_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_reduce.bias'.format(stage, block, op_idx)
						state_dict[se_cr_b_key].data[:] = state_dict_from_model[se_cr_b_key]
						se_ce_w_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_expand.weight'.format(stage, block, op_idx)
						state_dict[se_ce_w_key].data[index,:,:,:] = state_dict_from_model[se_ce_w_key]
						se_ce_b_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_expand.bias'.format(stage, block, op_idx)
						state_dict[se_ce_b_key].data[index] = state_dict_from_model[se_ce_b_key]
		del state_dict_from_model, index

		# shrink and expand
		if epoch >= args.search_epoch:
			logging.info('Now shrinking or expanding the arch')
			op_weights, depth_weights = get_op_and_depth_weights(model)
			parsed_arch = parse_architecture(op_weights, depth_weights)
			mc_num_dddict = get_mc_num_dddict(mc_mask_dddict)
			before_lat = get_lookup_latency(parsed_arch, mc_num_dddict, lat_lookup_key_dddict, lat_lookup)
			logging.info('Before, the current lat: {:.4f}, the target lat: {:.4f}'.format(before_lat, args.target_lat))
			num_stages = len(mc_mask_dddict)

			if before_lat > args.target_lat:
				logging.info('Shrinking......')
				stages = ['stage{}'.format(x) for x in range(1,num_stages+1)]
				mc_num_dddict, after_lat = fit_mc_num_by_latency(parsed_arch, mc_num_dddict, mc_maxnum_dddict, 
														lat_lookup_key_dddict, lat_lookup, args.target_lat, stages, sign=-1)
				for start in range(2,num_stages+1):
					stages = ['stage{}'.format(x) for x in range(start,num_stages+1)]
					mc_num_dddict, after_lat = fit_mc_num_by_latency(parsed_arch, mc_num_dddict, mc_maxnum_dddict, 
														lat_lookup_key_dddict, lat_lookup, args.target_lat, stages, sign=1)
			elif before_lat < args.target_lat:
				logging.info('Expanding......')
				stages = ['stage{}'.format(x) for x in range(1,num_stages+1)]
				mc_num_dddict, after_lat = fit_mc_num_by_latency(parsed_arch, mc_num_dddict, mc_maxnum_dddict, 
														lat_lookup_key_dddict, lat_lookup, args.target_lat, stages, sign=1)
				for start in range(2,num_stages+1):
					stages = ['stage{}'.format(x) for x in range(start,num_stages+1)]
					mc_num_dddict, after_lat = fit_mc_num_by_latency(parsed_arch, mc_num_dddict, mc_maxnum_dddict,
														lat_lookup_key_dddict, lat_lookup, args.target_lat, stages, sign=1)
			else:
				logging.info('No opeartion')
				after_lat = before_lat

			# change mc_mask_dddict based on mc_num_dddict
			for stage in parsed_arch:
				for block in parsed_arch[stage]:
					op_idx = parsed_arch[stage][block]
					if mc_num_dddict[stage][block][op_idx] != int(sum(mc_mask_dddict[stage][block][op_idx]).item()):
						mc_num = mc_num_dddict[stage][block][op_idx]
						max_mc_num = mc_mask_dddict[stage][block][op_idx].size(0)
						mc_mask_dddict[stage][block][op_idx].data[[True]*max_mc_num] = 0.0
						key = 'module.{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
						weight_copy = state_dict[key].clone().abs().cpu().numpy()
						weight_l1_norm = np.sum(weight_copy, axis=(1,2,3))
						weight_l1_order = np.argsort(weight_l1_norm)
						weight_l1_order_rev = weight_l1_order[::-1][:mc_num]
						mc_mask_dddict[stage][block][op_idx].data[weight_l1_order_rev.tolist()] = 1.0

			logging.info('After, the current lat: {:.4f}, the target lat: {:.4f}'.format(after_lat, args.target_lat))


		# save model
		model_path = os.path.join(args.save, 'searched_model_{:02}.pth.tar'.format(epoch+1))
		torch.save({
				'state_dict': state_dict,
				'mc_mask_dddict': mc_mask_dddict,
			}, model_path)


def train_wo_arch(train_queue, model, criterion, optimizer_w):
	objs_w = AverageMeter()

	model.train()

	for param in model.module.weight_parameters():
		param.requires_grad = True
	for param in model.module.arch_parameters():
		param.requires_grad = False

	steps = len(train_queue)

	for step, (x_w, target_w, _) in enumerate(train_queue):
		# print(x_w)
		
		x_w = x_w.cuda(non_blocking=True)
		target_w = target_w.cuda(non_blocking=True)

		logits_w_gumbel, _ = model(x_w, sampling=True, mode='gumbel')
		loss_w_gumbel_dist = criterion(logits_w_gumbel, target_w)
		loss_w_gumbel = sum(loss for loss in loss_w_gumbel_dist.values())

		# loss_dict_reduced = reduce_loss_dict(loss_w_gumbel_dist)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values()

		# reset switches of log_alphas
		model.module.reset_switches()

		optimizer_w.zero_grad()
		loss_w_gumbel.backward()
		if args.grad_clip > 0:
			nn.utils.clip_grad_norm_(model.module.weight_parameters(), args.grad_clip)
		optimizer_w.step()

		n = x_w.size(0)
		objs_w.update(loss_w_gumbel.item(), n)

		if step % args.print_freq == 0:
			logging.info('TRAIN wo_Arch Step: %04d/%d Objs: %f', step, steps, objs_w.avg)

	return objs_w.avg


def train_w_arch(train_queue, val_queue, model, criterion, optimizer_w, optimizer_a):
	objs_a = AverageMeter()
	objs_l = AverageMeter()
	objs_w = AverageMeter()

	model.train()
	steps = len(train_queue)

	for step, (x_w, target_w, _) in enumerate(train_queue):
		x_w = x_w.cuda(non_blocking=True)
		target_w = target_w.cuda(non_blocking=True)

		for param in model.module.weight_parameters():
			param.requires_grad = True
		for param in model.module.arch_parameters():
			param.requires_grad = False
		
		logits_w_gumbel, _ = model(x_w, sampling=True, mode='gumbel')
		loss_w_gumbel_dist = criterion(logits_w_gumbel, target_w)
		loss_w_gumbel = sum(loss for loss in loss_w_gumbel_dist.values())
		logits_w_random, _ = model(x_w, sampling=True, mode='random')
		loss_w_random_dist = criterion(logits_w_random, target_w)
		loss_w_random = sum(loss for loss in loss_w_random_dist.values())
		loss_w = loss_w_gumbel + loss_w_random

		optimizer_w.zero_grad()
		loss_w.backward()
		if args.grad_clip > 0:
			nn.utils.clip_grad_norm_(model.module.weight_parameters(), args.grad_clip)
		optimizer_w.step()

		n = x_w.size(0)
		objs_w.update(loss_w.item(), n)

		if step % 2 == 0:
			# optimize a
			try:
				x_a, target_a = next(val_queue_iter)
			except:
				val_queue_iter = iter(val_queue)
				x_a, target_a, _= next(val_queue_iter)

			x_a = x_a.cuda(non_blocking=True)
			target_a = target_a.cuda(non_blocking=True)

			for param in model.module.weight_parameters():
				param.requires_grad = False
			for param in model.module.arch_parameters():
				param.requires_grad = True

			logits_a, lat_dist = model(x_a, sampling=False)
			loss_a_dist = criterion(logits_a, target_a)
			loss_a = sum(loss for loss in loss_a_dist.values())
			lat = sum(loss for loss in lat_dist) # ???? why
			loss_l = torch.abs(lat / args.target_lat - 1.) * args.lambda_lat
			loss = loss_a + loss_l

			optimizer_a.zero_grad()
			loss.backward()
			if args.grad_clip > 0:
				nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
			optimizer_a.step()

			# ensure log_alphas to be a log probability distribution
			for log_alphas in model.module.arch_parameters():
				log_alphas.data = F.log_softmax(log_alphas.detach().data, dim=-1)

			n = x_a.size(0)
			objs_a.update(loss_a.item(), n)
			objs_l.update(loss_l.item(), n)

		if step % args.print_freq == 0:
			logging.info('TRAIN w_Arch Step: %04d/%d Objs_W: %f Objs_A: %f Objs_L: %f', 
						  step, steps, objs_w.avg, objs_a.avg, objs_l.avg)

	return [objs_w.avg, objs_a.avg]


def validate(val_queue, model, criterion, epoch, args):
	objs = AverageMeter()

	# model.eval()
	# disable moving average
	metric = SegmentationMetric(args.num_classes, args.distributed)
	metric.reset()

	torch.cuda.empty_cache()
	model.train()

	steps = len(val_queue)

	for step, (x, target, _) in enumerate(val_queue):
		x = x.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)
		with torch.no_grad():
			logits, _ = model(x, sampling=True, mode='gumbel')
			loss_dist = criterion(logits, target)
			loss = sum(los for los in loss_dist.values())
		# reset switches of log_alphas
		model.module.reset_switches()

		n = x.size(0)
		objs.update(loss.item(), n)
		metric.update(logits, target)
		pixAcc, mIoU = metric.get()

		if step % args.print_freq == 0:
			logging.info('VALIDATE Step: {:d}/{:d}, Objs: {:f} pixAcc: {:.3f}, mIoU: {:.3f}'.format(step, steps, objs.avg, pixAcc * 100, mIoU * 100))

	pixAcc, mIoU = metric.get()
	logging.info("[EVAL END] Epoch: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(epoch, pixAcc * 100, mIoU * 100))
	return [pixAcc * 100, mIoU * 100]


def get_lookup_latency(parsed_arch, mc_num_dddict, lat_lookup_key_dddict, lat_lookup):
	lat = lat_lookup['base']

	for stage in parsed_arch:
		for block in parsed_arch[stage]:
			op_idx = parsed_arch[stage][block]
			mid_channels_key = mc_num_dddict[stage][block][op_idx]
			lat_lookup_key = lat_lookup_key_dddict[stage][block][op_idx]
			lat += lat_lookup[lat_lookup_key][mid_channels_key]

	return lat


def fit_mc_num_by_latency(parsed_arch, mc_num_dddict, mc_maxnum_dddict, lat_lookup_key_dddict, lat_lookup, target_lat, stages, sign):
	# sign=1 for expand / sign=-1 for shrink
	assert sign == -1 or sign == 1
	lat = get_lookup_latency(parsed_arch, mc_num_dddict, lat_lookup_key_dddict, lat_lookup)

	parsed_mc_num_list = []
	parsed_mc_maxnum_list = []
	for stage in stages:
		for block in parsed_arch[stage]:
			op_idx = parsed_arch[stage][block]
			parsed_mc_num_list.append(mc_num_dddict[stage][block][op_idx])
			parsed_mc_maxnum_list.append(mc_maxnum_dddict[stage][block][op_idx])

	min_parsed_mc_num = min(parsed_mc_num_list)
	parsed_mc_ratio_list = [int(round(x/min_parsed_mc_num)) for x in parsed_mc_num_list]
	parsed_mc_bound_switches = [True] * len(parsed_mc_ratio_list)

	new_mc_num_dddict = copy.deepcopy(mc_num_dddict)
	new_lat = lat

	while any(parsed_mc_bound_switches) and (sign*new_lat <= sign*target_lat):
		mc_num_dddict = copy.deepcopy(new_mc_num_dddict)
		lat = new_lat
		list_idx = 0
		for stage in stages:
			for block in parsed_arch[stage]:
				op_idx = parsed_arch[stage][block]
				new_mc_num = mc_num_dddict[stage][block][op_idx] + sign * parsed_mc_ratio_list[list_idx]
				new_mc_num, switch = bound_clip(new_mc_num, parsed_mc_maxnum_list[list_idx])
				new_mc_num_dddict[stage][block][op_idx] = new_mc_num
				parsed_mc_bound_switches[list_idx] = switch
				list_idx += 1
		new_lat = get_lookup_latency(parsed_arch, new_mc_num_dddict, lat_lookup_key_dddict, lat_lookup)

	if sign == -1:
		mc_num_dddict = copy.deepcopy(new_mc_num_dddict)
		lat = new_lat

	return mc_num_dddict, lat


def bound_clip(mc_num, max_mc_num):
	min_mc_num = max_mc_num // 2

	if mc_num <= min_mc_num:
		new_mc_num = min_mc_num
		switch = False
	elif mc_num >= max_mc_num:
		new_mc_num = max_mc_num
		switch = False
	else:
		new_mc_num = mc_num
		switch = True

	return new_mc_num, switch


if __name__ == '__main__':
	start_time = time.time()
	main() 
	end_time = time.time()
	duration = end_time - start_time
	logging.info('Total searching time: %ds', duration)