import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import pickle

sys.path.append('..')

from tools.utils import measure_latency_in_ms
from models.layers import *
from models.seg_model_search import *
from models.seg_model_search import _ConvBNReLU

cudnn.enabled = True
cudnn.benchmark = True


PRIMITIVES = [
	'MBI_k3_e4',
	'MBI_k3_e8',
	'MBI_k5_e4',
	'MBI_k5_e8',
	'MBI_k3_e4_se',
	'MBI_k3_e8_se',
	'MBI_k5_e4_se',
	'MBI_k5_e8_se',
	# 'skip',
]

OPS = {
	'MBI_k3_e4' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e8' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e4' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e8' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k3_e4_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e8_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e4_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e8_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 5, s, affine=aff, act_func=act),
	# 'skip'      : lambda ic, mc, oc, s, aff, act: IdentityLayer(ic, oc),
}


def get_latency_lookup(is_cuda):
	latency_lookup = OrderedDict()

	print('LearningToDownsample, FeatureFusionModule, Classifer')
	block = LearningToDownsample(32, 48, 64)
	shape = (8, 3, 512, 1024) if is_cuda else (1, 3, 512, 1024)
	lat1  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	block = FeatureFusionModule(64, 144, 144)
	shape = [(8, 64, 64, 128),(8, 144, 16, 32)] if is_cuda else [(1, 64, 64, 128),(1, 144, 16, 32)]
	lat2  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	block = Classifer(144, 19)
	shape = (8, 144, 16, 32) if is_cuda else (1, 144, 16, 32)
	lat3  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	block = PyramidPooling(144)
	shape = (8, 144, 16, 32) if is_cuda else (1, 144, 16, 32)
	lat4  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	block = _ConvBNReLU(144 * 2, 144, 1)
	shape = (8, 144 * 2, 16, 32) if is_cuda else (1, 144 * 2, 16, 32)
	lat5  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	latency_lookup['base'] = lat1 + lat2 + lat3 + lat4 + lat5 # + 0.1  # 0.1 is the latency rectifier


	print('64x128 cin=64 cout=64 s=2 relu')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 64*4+1))
			# mc_list = list(range(0, 16*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 64*8+1))
			# mc_list = list(range(0, 16*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 64*2+1))
			# mc_list = list(range(0, 16*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](64, mc, 64, 2, True, 'relu')
			shape = (8, 64, 64, 128) if is_cuda else (1, 64, 64, 128)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_128_64_0_64_k{}_s2_relu'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_128_64_64_64_k{}_s2_relu'.format(block.name, block.kernel_size)
				else:
					key = '{}_128_64_128_64_k{}_s2_relu'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)


	print('32x64 cin=64 cout=64 s=1 relu')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 64*4+1))
			# mc_list = list(range(0, 16*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 64*8+1))
			# mc_list = list(range(0, 16*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 64*2+1))
			# mc_list = list(range(0, 16*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](64, mc, 64, 1, True, 'relu')
			shape = (8, 64, 32, 64) if is_cuda else (1, 64, 32, 64)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_64_64_0_64_k{}_s1_relu'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_64_64_64_64_k{}_s1_relu'.format(block.name, block.kernel_size)
				else:
					key = '{}_64_64_128_64_k{}_s1_relu'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)


	print('32x64 cin=64 cout=96 s=2 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 64*4+1))
			# mc_list = list(range(0, 24*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 64*8+1))
			# mc_list = list(range(0, 24*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 64*2+1))
			# mc_list = list(range(0, 24*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](64, mc, 96, 2, True, 'swish')
			shape = (8, 64, 32, 64) if is_cuda else (1, 64, 32, 64)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_64_64_0_96_k{}_s2_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_64_64_64_96_k{}_s2_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_64_64_128_96_k{}_s2_swish'.format(block.name, block.kernel_size)	
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)


	print('16x32 cin=96 cout=96 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 96*4+1))
			# mc_list = list(range(0, 40*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 96*8+1))
			# mc_list = list(range(0, 40*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 96*2+1))
			# mc_list = list(range(0, 40*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](96, mc, 96, 1, True, 'swish')
			shape = (8, 96, 16, 32) if is_cuda else (1, 96, 16, 32)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_32_96_0_96_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_32_96_96_96_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_32_96_192_96_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)


	print('16x32 cin=96 cout=128 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 96*4+1))
			# mc_list = list(range(0, 40*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 96*8+1))
			# mc_list = list(range(0, 40*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 96*2+1))
			# mc_list = list(range(0, 40*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](96, mc, 128, 1, True, 'swish')
			shape = (8, 96, 16, 32) if is_cuda else (1, 96, 16, 32)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_32_96_0_128_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_32_96_96_128_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_32_96_192_128_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 16x32 cin=128 cout=128 s=1 swish
	print('16x32 cin=128 cout=128 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 128*4+1))
			# mc_list = list(range(0, 80*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 128*8+1))
			# mc_list = list(range(0, 80*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 128*2+1))
			# mc_list = list(range(0, 80*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](128, mc, 128, 1, True, 'swish')
			shape = (16, 128, 16, 32) if is_cuda else (1, 128, 16, 32)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_32_128_0_128_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 ==0:
					key = '{}_32_128_128_128_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_32_128_256_128_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 16x32 cin=128 cout=144 s=1 swish
	print('16x32 cin=128 cout=144 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 128*4+1))
			# mc_list = list(range(0, 80*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 128*8+1))
			# mc_list = list(range(0, 80*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 128*2+1))
			# mc_list = list(range(0, 80*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](128, mc, 144, 1, True, 'swish')
			shape = (8, 128, 16, 32) if is_cuda else (1, 128, 16, 32)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_32_128_0_144_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_32_128_128_144_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_32_128_256_144_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 16x32 cin=144 cout=144 s=1 swish
	print('16x32 cin=144 cout=144 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 144*4+1))
			# mc_list = list(range(0, 112*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 144*8+1))
			# mc_list = list(range(0, 112*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 144*2+1))
			# mc_list = list(range(0, 112*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](144, mc, 144, 1, True, 'swish')
			shape = (16, 144, 16, 32) if is_cuda else (1, 144, 16, 32)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_32_144_0_144_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_32_144_144_144_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_32_144_288_144_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	return latency_lookup


# def convert_latency_lookup(latency_lookup):
# 	new_latency_lookup = OrderedDict()

# 	for key in latency_lookup:
# 		if key == 'base':
# 			new_latency_lookup['base'] = latency_lookup['base']
# 		else:
# 			mc_list  = list(latency_lookup[key].keys())
# 			lat_list = sorted(list(latency_lookup[key].values()))
# 			new_mc_list  = []
# 			new_lat_list = []
# 			for new_mc in range(1, mc_list[-1]+1):
# 				for idx in range(len(mc_list)):
# 					if new_mc == mc_list[idx]:
# 						new_mc_list.append(new_mc)
# 						new_lat_list.append(lat_list[idx])
# 						break
# 					if new_mc < mc_list[idx]:
# 						new_mc_list.append(new_mc)
# 						interval = (lat_list[idx] - lat_list[idx-1]) / (mc_list[idx] - mc_list[idx-1])
# 						new_lat = (new_mc - mc_list[idx-1]) * interval + lat_list[idx-1]
# 						new_lat_list.append(new_lat)
# 						break
# 			new_latency_lookup[key] = OrderedDict(list(zip(new_mc_list, new_lat_list)))

# 	return new_latency_lookup


if __name__ == '__main__':
	print('measure latency on gpu......')
	latency_lookup = get_latency_lookup(is_cuda=True)
	# latency_lookup = convert_latency_lookup(latency_lookup)
	with open('latency_gpu_fastscnn.pkl', 'wb') as f:
		pickle.dump(latency_lookup, f)

	print('measure latency on cpu......')
	latency_lookup = get_latency_lookup(is_cuda=False)
	# latency_lookup = convert_latency_lookup(latency_lookup)
	with open('latency_cpu_fastscnn.pkl', 'wb') as f:
		pickle.dump(latency_lookup, f)
