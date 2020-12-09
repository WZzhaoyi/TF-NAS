import os
import sys
import time
import glob
import logging
import argparse
import json
import numpy as np
import warnings
import datetime
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from tools.utils import AverageMeter, accuracy, SegmentationMetric
from tools.utils import count_parameters_in_MB
from tools.utils import create_exp_dir, save_checkpoint
from models.seg_model_eval import Network, NetworkCfg
from parsing_model import get_op_and_depth_weights
from parsing_model import parse_architecture
from parsing_model import get_mc_num_dddict
from dataset import get_segmentation_dataset
from dataset import IMAGENET_MEAN, IMAGENET_STD
from loss import MixSoftmaxCrossEntropyLoss


parser = argparse.ArgumentParser("training the searched architecture on imagenet")
# various path
parser.add_argument('--train_root', type=str, help='training image root path')
parser.add_argument('--val_root', type=str, help='validating image root path')
parser.add_argument('--train_list', type=str, help='training image list')
parser.add_argument('--val_list', type=str, help='validating image list')
parser.add_argument('--model_path', type=str, default='', help='the searched model path')
parser.add_argument('--config_path', type=str, default='', help='the model config path')
parser.add_argument('--save', type=str, default='./checkpoints/', help='model and log saving path')
parser.add_argument('--snapshot', type=str, default='', help='for reset')

# training hyper-parameters
parser.add_argument('--print_freq', type=float, default=100, help='print frequency')
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--epochs', type=int, default=250, help='num of total training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.2, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--num_classes', type=int, default=1000, help='class number of training set')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--drop_connect_rate', type=float, default=0.2, help='dropout connect rate')
parser.add_argument('--distributed', type=int, default=0, help='distributed')
parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')


args = parser.parse_args()

args.save = os.path.join(args.save, 'eval-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(logdir=os.path.join(args.save, 'log'))


class CrossEntropyLabelSmooth(nn.Module):
	def __init__(self, num_classes, epsilon):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, xs, targets):
		log_probs = self.logsoftmax(xs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (-targets * log_probs).mean(0).sum()
		return loss


def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)


def main():
	if not torch.cuda.is_available():
		logging.info('No GPU device available')
		sys.exit(1)
	set_seed(args.seed)
	cudnn.enabled=True
	cudnn.benchmark = True
	# torch.backends.cudnn.enabled = False
	logging.info("args = %s", args)

	# create model
	logging.info('parsing the architecture')
	if args.model_path and os.path.isfile(args.model_path):
		op_weights, depth_weights = get_op_and_depth_weights(args.model_path)
		parsed_arch = parse_architecture(op_weights, depth_weights)
		mc_mask_dddict = torch.load(args.model_path)['mc_mask_dddict']
		mc_num_dddict  = get_mc_num_dddict(mc_mask_dddict)
		model = Network(args.num_classes, parsed_arch, mc_num_dddict, None, args.dropout_rate, args.drop_connect_rate)
	elif args.config_path and os.path.isfile(args.config_path):
		model_config = json.load(open(args.config_path, 'r'))
		model = NetworkCfg(args.num_classes, model_config, None, args.dropout_rate, args.drop_connect_rate)
	else:
		raise Exception('invalid --model_path and --config_path')
	model = nn.DataParallel(model).cuda()
	# config = model.module.config
	# with open(os.path.join(args.save, 'model.config'), 'w') as f:
	# 	json.dump(config, f, indent=4)
	# logging.info(config)
	logging.info("param size = %fMB", count_parameters_in_MB(model))

	# define loss function (criterion) and optimizer
	criterion = MixSoftmaxCrossEntropyLoss(aux=False)
	criterion = criterion.cuda()
	# criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
	# criterion_smooth = criterion_smooth.cuda()

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	# define transform and initialize dataloader
	input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
	data_kwargs = {'transform': input_transform, 'base_size': 1024, 'crop_size': (512, 1024)}
	train_dataset = get_segmentation_dataset(split='train', mode='train', **data_kwargs)
	val_dataset = get_segmentation_dataset(split='val', mode='val', **data_kwargs)
	train_queue = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)

	val_queue = torch.utils.data.DataLoader(
		dataset=val_dataset,
		batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)

	# define learning rate scheduler
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
	best_acc_picAcc = 0
	best_acc_mIoU = 0
	start_epoch = 0

	# restart from snapshot
	if args.snapshot:
		logging.info('loading snapshot from {}'.format(args.snapshot))
		checkpoint = torch.load(args.snapshot)
		start_epoch = checkpoint['epoch']
		best_acc_picAcc = checkpoint['best_acc_picAcc']
		best_acc_mIoU = checkpoint['best_acc_mIoU']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), last_epoch=0)
		for epoch in range(start_epoch):
			current_lr = scheduler.get_lr()[0]
			logging.info('Epoch: %d lr %e', epoch, current_lr)
			if epoch < 5 and args.batch_size > 256:
				for param_group in optimizer.param_groups:
					param_group['lr'] = current_lr * (epoch + 1) / 5.0
				logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
			if epoch < 5 and args.batch_size > 256:
				for param_group in optimizer.param_groups:
					param_group['lr'] = current_lr
			scheduler.step()

	# the main loop
	for epoch in range(start_epoch, args.epochs):
		current_lr = scheduler.get_lr()[0]
		logging.info('Epoch: %d lr %e', epoch, current_lr)
		if epoch < 5 and args.batch_size > 256:
			for param_group in optimizer.param_groups:
				param_group['lr'] = current_lr * (epoch + 1) / 5.0
			logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)

		epoch_start = time.time()
		train_acc = train(train_queue, model, criterion, optimizer)
		logging.info('Train_acc: %f', train_acc)
		writer.add_scalar('Train/objs_w', train_acc, epoch)

		val_acc_picAcc, val_acc_mIoU, val_obj = validate(val_queue, model, criterion)
		logging.info('Val_acc_picAcc: %f', val_acc_picAcc)
		logging.info('Val_acc_mIoU: %f', val_acc_mIoU)
		writer.add_scalar('Val/pixAcc', val_acc_picAcc, epoch)
		writer.add_scalar('Val/mIoU', val_acc_mIoU, epoch)
		epoch_duration = time.time() - epoch_start
		eta_string = str(datetime.timedelta(seconds=int(epoch_duration*(args.epochs-epoch))))
		logging.info('Epoch time: %ds. Estimated time: %s', epoch_duration, eta_string)

		is_best = False
		if val_acc_mIoU > best_acc_mIoU:
			best_acc_picAcc = val_acc_picAcc
			best_acc_mIoU = val_acc_mIoU
			is_best = True
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_acc_picAcc': best_acc_picAcc,
			'best_acc_mIoU': best_acc_mIoU,
			'optimizer' : optimizer.state_dict(),
			}, is_best, args.save)

		if epoch < 5 and args.batch_size > 256:
			for param_group in optimizer.param_groups:
				param_group['lr'] = current_lr

		scheduler.step()


def train(train_queue, model, criterion, optimizer):
	objs = AverageMeter()
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	model.train()

	steps = len(train_queue)
	end = time.time()
	for step, data in enumerate(train_queue):
		data_time.update(time.time() - end)
		x = data[0].cuda(non_blocking=True)
		target = data[1].cuda(non_blocking=True)

		# forward
		batch_start = time.time()
		logits = model(x)
		loss_dict = criterion(logits, target)
		loss = sum(loss for loss in loss_dict.values())

		# backward
		optimizer.zero_grad()
		loss.backward()
		if args.grad_clip > 0:
			nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		optimizer.step()
		batch_time.update(time.time() - batch_start)

		n = x.size(0)
		objs.update(loss.data.item(), n)

		if step % args.print_freq == 0:
			duration = 0 if step == 0 else time.time() - duration_start
			duration_start = time.time()
			logging.info('TRAIN Step: %03d/%d Objs: %e Duration: %ds BTime: %.3fs DTime: %.4fs', 
									step, steps, objs.avg, duration, batch_time.avg, data_time.avg)
		end = time.time()

	return objs.avg


def validate(val_queue, model, criterion):
	objs = AverageMeter()
	model.eval()
	metric = SegmentationMetric(args.num_classes, args.distributed)
	metric.reset()

	steps = len(val_queue)
	for step, data in enumerate(val_queue):
		x = data[0].cuda(non_blocking=True)
		target = data[1].cuda(non_blocking=True)

		with torch.no_grad():
			logits = model(x)
			loss_dict = criterion(logits, target)
			loss = sum(loss for loss in loss_dict.values())

		n = x.size(0)
		objs.update(loss.data.item(), n)
		metric.update(logits, target)
		pixAcc, mIoU = metric.get()

		if step % args.print_freq == 0:
			duration = 0 if step == 0 else time.time() - duration_start
			duration_start = time.time()
			logging.info('VALID Step: %03d/%d Objs: %e pixAcc: %f mIoU: %f Duration: %ds', step, steps, objs.avg, pixAcc * 100, mIoU * 100, duration)
	pixAcc, mIoU = metric.get()
	return pixAcc * 100, mIoU * 100, objs.avg


if __name__ == '__main__':
	main()
