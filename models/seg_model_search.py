import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MBInvertedResBlock, IdentityLayer
from collections import OrderedDict

PRIMITIVES = [
	'MBI_k3_e3',
	'MBI_k3_e6',
	'MBI_k5_e3',
	'MBI_k5_e6',
	'MBI_k3_e3_se',
	'MBI_k3_e6_se',
	'MBI_k5_e3_se',
	'MBI_k5_e6_se',
	# 'skip',
]

OPS = {
	'MBI_k3_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k3_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e6_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e6_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 5, s, affine=aff, act_func=act),
	# 'skip'      : lambda ic, mc, oc, s, aff, act: IdentityLayer(ic, oc),
}


class MixedOP(nn.Module):
	def __init__(self, in_channels, out_channels, stride, affine, act_func, num_ops, mc_num_dict, lat_lookup):
		super(MixedOP, self).__init__()
		self.num_ops = num_ops
		self.lat_lookup = lat_lookup
		self.mc_num_dict = mc_num_dict
		self.m_ops = nn.ModuleList()

		for i in range(num_ops):
			primitive = PRIMITIVES[i]
			mid_channels = self.mc_num_dict[i]
			op = OPS[primitive](in_channels, mid_channels, out_channels, stride, affine, act_func)
			self.m_ops.append(op)

		self._initialize_log_alphas()
		self.reset_switches()

	def fink_ori_idx(self, idx):
		count = 0
		for ori_idx in range(len(self.switches)):
			if self.switches[ori_idx]:
				count += 1
				if count == (idx + 1):
					break
		return ori_idx

	def forward(self, x, sampling, mode):
		if sampling:
			weights = self.log_alphas[self.switches]
			if mode == 'gumbel':
				weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
				idx = torch.argmax(weights).item()
				self.switches[idx] = False
			elif mode == 'gumbel_2':
				weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
				idx = torch.argmax(weights).item()
				idx = self.fink_ori_idx(idx)
				self.reset_switches()
			elif mode == 'min_alphas':
				idx = torch.argmin(weights).item()
				idx = self.fink_ori_idx(idx)
				self.reset_switches()
			elif mode == 'max_alphas':
				idx = torch.argmax(weights).item()
				idx = self.fink_ori_idx(idx)
				self.reset_switches()
			elif mode == 'random':
				idx = random.choice(range(len(weights)))
				idx = self.fink_ori_idx(idx)
				self.reset_switches()
			else:
				raise ValueError('invalid sampling mode...')
			op = self.m_ops[idx]
			return op(x), 0
		else:
			weights = F.gumbel_softmax(self.log_alphas, self.T, hard=False)
			lats = self.get_lookup_latency(x.size(-1))
			out = sum(w*op(x) for w, op in zip(weights, self.m_ops))
			out_lat = sum(w*lat for w, lat in zip(weights, lats))
			return out, out_lat

	def get_lookup_latency(self, size):
		lats = []
		for idx, op in enumerate(self.m_ops):
			if isinstance(op, IdentityLayer):
				lats.append(0)
			else:
				key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												op.name,
												size,
												op.in_channels,
												op.se_channels,
												op.out_channels,
												op.kernel_size,
												op.stride,
												op.act_func)
				mid_channels = op.mid_channels
				lats.append(self.lat_lookup[key][mid_channels])

		return lats

	def _initialize_log_alphas(self):
		alphas = torch.zeros((self.num_ops,))
		log_alphas = F.log_softmax(alphas, dim=-1)
		self.register_parameter('log_alphas', nn.Parameter(log_alphas))

	def reset_switches(self):
		self.switches = [True] * self.num_ops

	def set_temperature(self, T):
		self.T = T


class MixedStage(nn.Module):
	def __init__(self, ics, ocs, ss, affs, acts, mc_num_ddict, lat_lookup, stage_type):
		super(MixedStage, self).__init__()
		self.lat_lookup = lat_lookup
		self.mc_num_ddict = mc_num_ddict
		self.stage_type = stage_type # 0 for stage6 || 1 for stage1 || 2 for stage2 || 3 for stage3/4/5
		self.start_res = 0 if ((ics[0] == ocs[0]) and (ss[0] == 1)) else 1
		self.num_res = len(ics) - self.start_res + 1

		# stage6
		if stage_type == 0:
			self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES), mc_num_ddict['block1'], lat_lookup)
		# stage1
		elif stage_type == 1:
			self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES), mc_num_ddict['block1'], lat_lookup)
			self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES), mc_num_ddict['block2'], lat_lookup)
		# stage2
		elif stage_type == 2:
			self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES), mc_num_ddict['block1'], lat_lookup)
			self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES), mc_num_ddict['block2'], lat_lookup)
			self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES), mc_num_ddict['block3'], lat_lookup)
		# stage3, stage4, stage5
		elif stage_type == 3:
			self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES), mc_num_ddict['block1'], lat_lookup)
			self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES), mc_num_ddict['block2'], lat_lookup)
			self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES), mc_num_ddict['block3'], lat_lookup)
			self.block4 = MixedOP(ics[3], ocs[3], ss[3], affs[3], acts[3], len(PRIMITIVES), mc_num_ddict['block4'], lat_lookup)
		else:
			raise ValueError('invalid stage_type...')

		self._initialize_betas()

	def forward(self, x, sampling, mode):
		res_list = [x,]
		lat_list = [0.,]

		# stage6
		if self.stage_type == 0:
			out1, lat1 = self.block1(x, sampling, mode)
			res_list.append(out1)
			lat_list.append(lat1)
		# stage1
		elif self.stage_type == 1:
			out1, lat1 = self.block1(x, sampling, mode)
			res_list.append(out1)
			lat_list.append(lat1)
			out2, lat2 = self.block2(out1, sampling, mode)
			res_list.append(out2)
			lat_list.append(lat1+lat2)
		# stage2
		elif self.stage_type == 2:
			out1, lat1 = self.block1(x, sampling, mode)
			res_list.append(out1)
			lat_list.append(lat1)
			out2, lat2 = self.block2(out1, sampling, mode)
			res_list.append(out2)
			lat_list.append(lat1+lat2)
			out3, lat3 = self.block3(out2, sampling, mode)
			res_list.append(out3)
			lat_list.append(lat1+lat2+lat3)
		# stage3, stage4, stage5
		elif self.stage_type == 3:
			out1, lat1 = self.block1(x, sampling, mode)
			res_list.append(out1)
			lat_list.append(lat1)
			out2, lat2 = self.block2(out1, sampling, mode)
			res_list.append(out2)
			lat_list.append(lat1+lat2)
			out3, lat3 = self.block3(out2, sampling, mode)
			res_list.append(out3)
			lat_list.append(lat1+lat2+lat3)
			out4, lat4 = self.block4(out3, sampling, mode)
			res_list.append(out4)
			lat_list.append(lat1+lat2+lat3+lat4)
		else:
			raise ValueError

		weights = F.softmax(self.betas, dim=-1)
		out = sum(w*res for w, res in zip(weights, res_list[self.start_res:]))
		out_lat = sum(w*lat for w, lat in zip(weights, lat_list[self.start_res:]))

		return out, out_lat

	def _initialize_betas(self):
		betas = torch.zeros((self.num_res))
		self.register_parameter('betas', nn.Parameter(betas))


class Network(nn.Module):
	def __init__(self, num_classes, mc_num_dddict, lat_lookup):
		super(Network, self).__init__()
		self.lat_lookup = lat_lookup
		self.mc_num_dddict = mc_num_dddict
		self.nclass = num_classes
		self.first_stem  = LearningToDownsample(32, 48, 64)

		self.stage1 = MixedStage(
							ics  = [64,80],
							ocs  = [80,80],
							ss   = [2,1],
							affs = [False, False],
							acts = ['relu', 'relu'],
							mc_num_ddict = mc_num_dddict['stage1'],
							lat_lookup = lat_lookup,
							stage_type = 1,)
		self.stage2 = MixedStage(
							ics  = [80,96,96],
							ocs  = [96,96,96],
							ss   = [2,1,1],
							affs = [False, False, False],
							acts = ['swish', 'swish', 'swish'],
							mc_num_ddict = mc_num_dddict['stage2'],
							lat_lookup = lat_lookup,
							stage_type = 2,)
		self.stage3 = MixedStage(
							ics  = [96,128,128,128],
							ocs  = [128,128,128,128],
							ss   = [1,1,1,1],
							affs = [False, False, False, False],
							acts = ['swish', 'swish', 'swish', 'swish'],
							mc_num_ddict = mc_num_dddict['stage3'],
							lat_lookup = lat_lookup,
							stage_type = 3,)
		self.stage4 = MixedStage(
							ics  = [128,144,144,144],
							ocs  = [144,144,144,144],
							ss   = [1,1,1,1],
							affs = [False, False, False, False],
							acts = ['swish', 'swish', 'swish', 'swish'],
							mc_num_ddict = mc_num_dddict['stage4'],
							lat_lookup = lat_lookup,
							stage_type = 3,)
		self.stage5 = MixedStage(
							ics  = [144,192,192,192],
							ocs  = [192,192,192,192],
							ss   = [1,1,1,1],
							affs = [False, False, False, False],
							acts = ['swish', 'swish', 'swish', 'swish'],
							mc_num_ddict = mc_num_dddict['stage5'],
							lat_lookup = lat_lookup,
							stage_type = 3,)
		self.stage6 = MixedStage(
							ics  = [192,],
							ocs  = [320,],
							ss   = [1,],
							affs = [False,],
							acts = ['swish',],
							mc_num_ddict = mc_num_dddict['stage6'],
							lat_lookup = lat_lookup,
							stage_type = 0,)
		self.feature_mix_layer = FeatureFusionModule(64, 320, 320)
		self.classifier = Classifer(320, self.nclass)

		self._initialization()

	def forward(self, x, sampling, mode='max'):
		size = x.size()[2:]
		out_lat = self.lat_lookup['base'] if not sampling else 0.0
		x = self.first_stem(x)
		higher_res_features = x

		x, lat = self.stage1(x, sampling, mode)
		out_lat += lat
		x, lat = self.stage2(x, sampling, mode)
		out_lat += lat
		x, lat = self.stage3(x, sampling, mode)
		out_lat += lat
		x, lat = self.stage4(x, sampling, mode)
		out_lat += lat
		x, lat = self.stage5(x, sampling, mode)
		out_lat += lat
		x, lat = self.stage6(x, sampling, mode)
		out_lat += lat

		x = self.feature_mix_layer(higher_res_features, x)
		x = self.classifier(x)
		x = F.interpolate(x, size, mode='bilinear', align_corners=True)

		return x, out_lat

	def set_temperature(self, T):
		for m in self.modules():
			if isinstance(m, MixedOP):
				m.set_temperature(T)

	def weight_parameters(self):
		_weight_parameters = []

		for k, v in self.named_parameters():
			if not (k.endswith('log_alphas') or k.endswith('betas')):
				_weight_parameters.append(v)
		
		return _weight_parameters

	def arch_parameters(self):
		_arch_parameters = []

		for k, v in self.named_parameters():
			if k.endswith('log_alphas') or k.endswith('betas'):
				_arch_parameters.append(v)

		return _arch_parameters

	def log_alphas_parameters(self):
		_log_alphas_parameters = []

		for k, v in self.named_parameters():
			if k.endswith('log_alphas'):
				_log_alphas_parameters.append(v)

		return _log_alphas_parameters

	def betas_parameters(self):
		_betas_parameters = []

		for k, v in self.named_parameters():
			if k.endswith('betas'):
				_betas_parameters.append(v)

		return _betas_parameters

	def reset_switches(self):
		for m in self.modules():
			if isinstance(m, MixedOP):
				m.reset_switches()

	def _initialization(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, norm_layer=nn.BatchNorm2d):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = SeparableConv2d(dw_channels1, dw_channels2, stride=2, relu_first=False, norm_layer=norm_layer)
        self.dsconv2 = SeparableConv2d(dw_channels2, out_channels, stride=2, relu_first=False, norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, norm_layer=nn.BatchNorm2d):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _ConvBNReLU(lower_in_channels, out_channels, 1, norm_layer=norm_layer)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            norm_layer(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            norm_layer(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, norm_layer=nn.BatchNorm2d):
        super(Classifer, self).__init__()
        self.dsconv1 = SeparableConv2d(dw_channels, dw_channels, stride=stride, relu_first=False,
                                       norm_layer=norm_layer)
        self.dsconv2 = SeparableConv2d(dw_channels, dw_channels, stride=stride, relu_first=False,
                                       norm_layer=norm_layer)
        self.conv = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x
