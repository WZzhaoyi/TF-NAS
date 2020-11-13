import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import time
from torch import distributed as dist



def measure_latency_in_ms(model, input_shape, is_cuda):
	INIT_TIMES = 10
	LAT_TIMES  = 100
	lat = AverageMeter()
	model.eval()
	if(isinstance(input_shape,list)):
		x = [torch.randn(item) for item in input_shape]
	else:
		x = torch.randn(input_shape)
	if is_cuda:
		model = model.cuda()
		if(isinstance(input_shape,list)):
			x = [item.cuda() for item in x]
			batch_size = x[0].size(0)
		else:
			x = x.cuda()
			batch_size =x.size(0)
	else:
		model = model.cpu()
		if(isinstance(input_shape,list)):
			x = [item.cpu() for item in x]
		else:
			x = x.cpu()
		batch_size = 1
	with torch.no_grad():
		for _ in range(INIT_TIMES):
			if(isinstance(x,list) and len(x) == 2):
				output = model(x[0],x[1])
			else:
				output = model(x)

		for _ in range(LAT_TIMES):
			tic = time.time()
			if(isinstance(x,list) and len(x) == 2):
				output = model(x[0],x[1])
			else:
				output = model(x)
			toc = time.time()
			lat.update(toc-tic, batch_size)

	return lat.avg * 1000 # save as ms


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def drop_connect(x, training=False, drop_connect_rate=0.0):
	"""Apply drop connect."""
	if not training:
		return x
	keep_prob = 1 - drop_connect_rate
	random_tensor = keep_prob + torch.rand(
		(x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
	random_tensor.floor_()  # binarize
	output = x.div(keep_prob) * random_tensor
	return output


def channel_shuffle(x, groups):
	assert groups > 1
	batchsize, num_channels, height, width = x.size()
	assert (num_channels % groups == 0)
	channels_per_group = num_channels // groups
	# reshape
	x = x.view(batchsize, groups, channels_per_group, height, width)
	# transpose
	x = torch.transpose(x, 1, 2).contiguous()
	# flatten
	x = x.view(batchsize, -1, height, width)
	return x


def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: {}'.format(kernel_size)
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2


def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
	filename = os.path.join(save, 'checkpoint.pth.tar')
	torch.save(state, filename)
	if is_best:
		best_filename = os.path.join(save, 'model_best.pth.tar')
		shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))

	if scripts_to_save is not None:
		os.makedirs(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, distributed):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.distributed = distributed
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def reduce_tensor(tensor):
            rt = tensor.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            return rt

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)
            if self.distributed:
                correct = reduce_tensor(correct)
                labeled = reduce_tensor(labeled)
                inter = reduce_tensor(inter.cuda())
                union = reduce_tensor(union.cuda())
            torch.cuda.synchronize()
            self.total_correct += correct.item()
            self.total_label += labeled.item()
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self, return_category_iou=False):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        if return_category_iou:
            return pixAcc, mIoU, IoU.cpu().numpy()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0)#.item()
    pixel_correct = torch.sum((predict == target) * (target > 0))#.item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()

