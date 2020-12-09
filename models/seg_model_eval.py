import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

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


class Network(nn.Module):
	def __init__(self, num_classes, parsed_arch, mc_num_dddict, lat_lookup=None, dropout_rate=0.0, drop_connect_rate=0.0):
		super(Network, self).__init__()
		self.lat_lookup = lat_lookup
		self.mc_num_dddict = mc_num_dddict
		self.parsed_arch = parsed_arch
		self.dropout_rate = dropout_rate
		self.drop_connect_rate = drop_connect_rate
		self.block_count = self._get_block_count()
		self.block_idx = 0
		self.num_classes = num_classes

		self.first_stem  = LearningToDownsample(32, 48, 64)
		
		self.stage1 = self._make_stage('stage1',
							ics  = [64,64,64],
							ocs  = [64,64,64],
							ss   = [2,1,1],
							affs = [False, False, False],
							acts = ['relu', 'relu', 'relu'],)
		self.stage2 = self._make_stage('stage2',
							ics  = [64,96,96],
							ocs  = [96,96,96],
							ss   = [2,1,1],
							affs = [False, False, False],
							acts = ['swish', 'swish', 'swish'],)
		self.stage3 = self._make_stage('stage3',
							ics  = [96,128,128],
							ocs  = [128,128,128],
							ss   = [1,1,1],
							affs = [False, False, False],
							acts = ['swish', 'swish', 'swish'],)
		self.stage4 = self._make_stage('stage4',
							ics  = [128,144,144],
							ocs  = [144,144,144],
							ss   = [1,1,1],
							affs = [False, False, False],
							acts = ['swish', 'swish', 'swish'],)
		
		self.ppm = PyramidPooling(144)
		self.out = _ConvBNReLU(144 * 2, 144, 1)
		self.feature_mix_layer = FeatureFusionModule(64, 144, 144)
		self.classifier = Classifer(144, self.num_classes)

		self._initialization()

	def _get_block_count(self):
		count = 1
		for stage in self.parsed_arch:
			count += len(self.parsed_arch[stage])

		return count

	def _make_stage(self, stage_name, ics, ocs, ss, affs, acts):
		stage = nn.ModuleList()
		for i, block_name in enumerate(self.parsed_arch[stage_name]):
			self.block_idx += 1
			op_idx = self.parsed_arch[stage_name][block_name]
			primitive = PRIMITIVES[op_idx]
			mc = self.mc_num_dddict[stage_name][block_name][op_idx]
			op = OPS[primitive](ics[i], mc, ocs[i], ss[i], affs[i], acts[i])
			op.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
			stage.append(op)

		return stage

	def forward(self, x):
		size = x.size()[2:]
		x = self.first_stem(x)
		higher_res_features = x

		for block in self.stage1:
			x = block(x)
		for block in self.stage2:
			x = block(x)
		for block in self.stage3:
			x = block(x)
		for block in self.stage4:
			x = block(x)
		x = self.ppm(x)
		x = self.out(x)
		lower_res_features = x

		x = self.feature_mix_layer(higher_res_features,lower_res_features)
		if self.dropout_rate > 0.0:
			x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.classifier(x)
		x = F.interpolate(x, size, mode='bilinear', align_corners=True)
		
		return x

	def get_lookup_latency(self, x):
		if not self.lat_lookup:
			return 0.0

		lat = self.lat_lookup['base']
		x = self.first_stem(x)
		x = self.second_stem(x)

		for block in self.stage1:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage2:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage3:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage4:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage5:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage6:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)

		return lat

	@property
	def config(self):
		return {
			'first_stem':  self.first_stem.config,
			'stage1': [block.config for block in self.stage1],
			'stage2': [block.config for block in self.stage2],
			'stage3': [block.config for block in self.stage3],
			'stage4': [block.config for block in self.stage4],
			'ppm': self.ppm,
			'out': self.out,
			'feature_mix_layer': self.feature_mix_layer.config,
			'classifier': self.classifier.config,
		}

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


class NetworkCfg(nn.Module):
	def __init__(self, num_classes, model_config, lat_lookup=None, dropout_rate=0.0, drop_connect_rate=0.0):
		super(NetworkCfg, self).__init__()
		self.lat_lookup = lat_lookup
		self.model_config = model_config
		self.dropout_rate = dropout_rate
		self.drop_connect_rate = drop_connect_rate
		self.block_count = self._get_block_count()
		self.block_idx = 0

		self.first_stem  = set_layer_from_config(model_config['first_stem'])
		self.second_stem = set_layer_from_config(model_config['second_stem'])
		self.block_idx += 1
		self.second_stem.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
		self.stage1 = self._make_stage('stage1')
		self.stage2 = self._make_stage('stage2')
		self.stage3 = self._make_stage('stage3')
		self.stage4 = self._make_stage('stage4')
		self.stage5 = self._make_stage('stage5')
		self.stage6 = self._make_stage('stage6')
		self.feature_mix_layer = set_layer_from_config(model_config['feature_mix_layer'])
		self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

		classifier_config = model_config['classifier']
		classifier_config['out_features'] = num_classes
		self.classifier = set_layer_from_config(classifier_config)

		self._initialization()

	def _get_block_count(self):
		count = 1
		for k in self.model_config.keys():
			if k.startswith('stage'):
				count += len(self.model_config[k])

		return count

	def _make_stage(self, stage_name):
		stage = nn.ModuleList()
		for layer_config in self.model_config[stage_name]:
			self.block_idx += 1
			layer = set_layer_from_config(layer_config)
			layer.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
			stage.append(layer)

		return stage

	def forward(self, x):
		x = self.first_stem(x)
		x = self.second_stem(x)

		for block in self.stage1:
			x = block(x)
		for block in self.stage2:
			x = block(x)
		for block in self.stage3:
			x = block(x)
		for block in self.stage4:
			x = block(x)
		for block in self.stage5:
			x = block(x)
		for block in self.stage6:
			x = block(x)

		x = self.feature_mix_layer(x)
		x = self.global_avg_pooling(x)
		x = x.view(x.size(0), -1)
		if self.dropout_rate > 0.0:
			x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.classifier(x)
		
		return x

	def get_lookup_latency(self, x):
		if not self.lat_lookup:
			return 0.0

		lat = self.lat_lookup['base']
		x = self.first_stem(x)
		x = self.second_stem(x)

		for block in self.stage1:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage2:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage3:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage4:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage5:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage6:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)

		return lat

	@property
	def config(self):
		return {
			'first_stem':  self.first_stem.config,
			'second_stem': self.second_stem.config,
			'stage1': [block.config for block in self.stage1],
			'stage2': [block.config for block in self.stage2],
			'stage3': [block.config for block in self.stage3],
			'stage4': [block.config for block in self.stage4],
			'stage5': [block.config for block in self.stage5],
			'stage6': [block.config for block in self.stage6],
			'feature_mix_layer': self.feature_mix_layer.config,
			'classifier': self.classifier.config,
		}

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


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        for size in sizes:
            self.avgpools.add_module('stage' + str(size), nn.AdaptiveAvgPool2d(size))
            self.convs.add_module('stage' + str(size), _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer, **kwargs))

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for (avgpool, conv) in zip(self.avgpools, self.convs):
            feats.append(F.interpolate(conv(avgpool(x)), size, mode='bilinear', align_corners=True))
        return torch.cat(feats, dim=1)


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
        self.conv_lower_res = nn.Sequential(OrderedDict([
			('conv1',nn.Conv2d(out_channels, out_channels, 1)),
			('norm',norm_layer(out_channels))
		]))
        self.conv_higher_res = nn.Sequential(OrderedDict([
			('conv1',nn.Conv2d(highter_in_channels, out_channels, 1)),
			('norm',norm_layer(out_channels))
		]))
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
        self.conv = nn.Sequential(OrderedDict([
			('dropout', nn.Dropout2d(0.1)),
			('conv1', nn.Conv2d(dw_channels, num_classes, 1))
		]))

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x
