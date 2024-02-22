"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import load_dataset.data_loader_Colorectal as data_loader
import os
import numpy as np
import math
import copy
from datetime import datetime
import multiprocessing
from utils_ghostv2 import Utils
from torchprofile import profile_macs
from template.drop import drop_path
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score, balanced_accuracy_score

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModuleV2, self).__init__()
        self.gate_fn = nn.Sigmoid()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.short_conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=out.shape[-1], mode='nearest')


class GhostBottleneckV2(nn.Module):

    def __init__(self, in_chs, out_chs, expantion=3, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        mid_chs = in_chs * expantion

        # Point-wise expansion
        self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True)

            # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()

class SELayer(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super(SELayer, self).__init__()
        reduce_chs = max(1, int(in_chs * se_ratio))
        self.act_fn = F.relu
        self.gate_fn = sigmoid
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, expansion=3, stride=1, dilation=1, act_func='h_swish', attention=False, drop_connect_rate=0.0, dense=False, affine=True):
        super(MBConv, self).__init__()
        interChannels = expansion * C_out
        self.op1 = nn.Sequential(
            nn.Conv2d(C_in, interChannels, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(interChannels, affine=affine)
        )
        self.op2 = nn.Sequential(
                   nn.Conv2d(interChannels, interChannels, kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) / 2) * dilation, bias=False, dilation=dilation, groups=interChannels),
                   nn.BatchNorm2d(interChannels, affine=affine)
        )
        self.op3 = nn.Sequential(
            nn.Conv2d(interChannels, C_out, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(C_out, affine=affine)
        )

        if act_func == 'relu':
            self.act_func = nn.ReLU(inplace=True)
        else:
            self.act_func = Hswish(inplace=True)
        if attention:
            self.se = CoordAtt(interChannels, interChannels)
        else:
            self.se = nn.Sequential()
        self.drop_connect_rate = drop_connect_rate
        self.stride = stride
        self.dense = int(dense)
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, x):
        out = self.op1(x)
        out = self.act_func(out)
        out = self.op2(out)
        out = self.act_func(out)
        out = self.se(out)
        out = self.op3(out)

        if self.drop_connect_rate > 0:
            out = drop_path(out, drop_prob=self.drop_connect_rate, training=self.training)
        if self.stride == 1 and self.dense:
            out = torch.cat([x, out], dim=1)
        elif self.stride == 1 and self.C_in == self.C_out:
            out = out + x
        return out

class DenseBlock(nn.Module):
    def __init__(self, layer_types, in_channels, out_channels, kernel_sizes, expansions, strides, act_funcs, attentions, drop_connect_rates, dense):
        super(DenseBlock, self).__init__()
        self.layer_types = list(map(int, layer_types.split()))
        self.in_channels = list(map(int, in_channels.split()))
        self.out_channels = list(map(int, out_channels.split()))
        self.kernel_sizes = list(map(int, kernel_sizes.split()))
        self.expansions = list(map(int, expansions.split()))
        self.attentions = list(map(bool, map(int, attentions.split())))
        self.strides = list(map(int, strides.split()))
        self.act_funcs = list(map(str, act_funcs.split()))
        self.drop_connect_rates = list(map(float, drop_connect_rates.split()))
        self.dense = int(dense)

        self.layer = self._make_dense(len(self.out_channels))

    def _make_dense(self, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            if self.layer_types[i] == 0:
                layers.append(Identity())
            else:
                layers.append(MBConv(self.in_channels[i], self.out_channels[i], self.kernel_sizes[i], self.expansions[i], self.strides[i], 1, self.act_funcs[i], self.attentions[i], self.drop_connect_rates[i], self.dense))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.layer(out)
        return out

class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.Hswish = Hswish(inplace=True)
        #generated_init


    def forward(self, x):
        out_aux = None
        #generate_forward

        out = self.evolved_block2(out)
        #out = self.Hswish(self.bn_end1(self.conv_end1(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.Hswish(self.conv_end1(out))
        out = out.view(out.size(0), -1)
        # out = self.Hswish(self.dropout(self.linear1(out)))

        out = F.dropout(out, p=0.2, training=self.training)
        out = self.linear(out)

        return out, out_aux


class TrainModel(object):
    def __init__(self, is_test, particle, batch_size, weight_decay):
        if is_test:
            full_trainloader = data_loader.get_train_loader('../datasets/Colorectal_data/train', batch_size=batch_size, augment=True,shuffle=True, num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader('../datasets/Colorectal_data/val', batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader
        else:
            trainloader, validate_loader = data_loader.get_train_valid_loader('../datasets/Colorectal_data/train', batch_size=16, augment=True, subset_size=1,valid_size=0.1, shuffle=True,num_workers=4, pin_memory=True)
            self.trainloader = trainloader
            self.validate_loader = validate_loader

        net = EvoCNNModel()
        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        self.flops = 10e9
        self.best_epoch = 0
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.weight_decay = weight_decay
        self.particle = copy.deepcopy(particle)

        if not is_test:
            self.log_record('loading weights from the weight pool', first_time=True)
        else:
            self.log_record('testing, not need to load weights from the weight pool', first_time=True)
        self.net = self.net.cuda()

    # def load_weights_from_pool(self):
    #     pool_path = './trained_models/'
    #     subparticle_length = Utils.get_params('PSO', 'particle_length') // 3
    #     subParticles = [self.particle[0:subparticle_length], self.particle[subparticle_length:2 * subparticle_length],
    #                     self.particle[2 * subparticle_length:]]
    #     weight_pool = self.get_flist(pool_path)
    #     print(weight_pool)
    #
    #     for op_name in ['begin1', 'end1', 'linear', 'tranLayer0', 'tranLayer1']:
    #         is_found = False
    #         for op in weight_pool:
    #             if op_name in op:
    #                 is_found = True
    #                 self.log_record('Inheriting the weights of '+op_name + ' from the weight pool.')
    #                 # op_weights = torch.load(pool_path+op, map_location='cuda')
    #                 op_weights = torch.load(pool_path + op, map_location=torch.device('cpu'))
    #                 adjusted_op_weights = self.adjust_weights(op_name, op_weights)
    #                 self.net.load_state_dict(adjusted_op_weights, strict=False)
    #         if not is_found:
    #             self.log_record(op_name+' is not found in the weight pool.')
    #
    #     for j, subParticle in enumerate(subParticles):
    #         for number, dimen in enumerate(subParticle):
    #             type = int(dimen)
    #             if not type == 0:
    #                 is_found = False
    #                 op_name = 'evolved_block'+str(j)+'.layer.'+str(number)
    #                 for op in weight_pool:
    #                     if op_name+'-'+str(type)+'-' in op:
    #                         is_found = True
    #                         self.log_record('Inheriting the weights of ' + op_name+'-'+str(type) + ' from the weight pool.')
    #                         # op_weights = torch.load(pool_path + op, map_location='cuda')
    #                         op_weights = torch.load(pool_path + op, map_location=torch.device('cpu'))
    #                         adjusted_op_weights = self.adjust_weights(op_name, op_weights)
    #                         self.net.load_state_dict(adjusted_op_weights, strict=False)
    #                 if not is_found:
    #                     self.log_record(op_name+'-'+str(type)+' is not found in the weight pool.')

    # def adjust_weights(self, op_name, op_weights):
    #     net_items = self.net.state_dict().items()
    #     adjusted_op_weights = {}
    #
    #     for curr_key, curr_val in net_items:
    #         if op_name+'.' in curr_key:
    #             curr_val = curr_val.numpy()
    #             pool_weight = op_weights[curr_key].numpy()
    #             shape_pool_op = np.shape(pool_weight)
    #             shape_curr_op = np.shape(curr_val)
    #             shape_inherit = np.min([shape_pool_op, shape_curr_op], axis=0)
    #             _, subset_inherit = self.get_subset(pool_weight, shape_inherit)
    #             adjusted_op_weight = self.inherit_weight(curr_val, subset_inherit)
    #             adjusted_op_weights[curr_key] = torch.from_numpy(adjusted_op_weight)
    #     return adjusted_op_weights

    # def get_subset(self, a, bshape):
    #     slices = []
    #     for i, dim in enumerate(a.shape):
    #         center = dim // 2
    #         start = center - bshape[i] // 2
    #         stop = start + bshape[i]
    #         slices.append(slice(start, stop))
    #     return slices, a[tuple(slices)]

    # def inherit_weight(self, curr_val, subset_inherit):
    #     inherit_shape = np.shape(subset_inherit)
    #     slices = []
    #     # curr_val = copy.deepcopy(curr_val)
    #     for i, dim in enumerate(curr_val.shape):
    #         center = dim // 2
    #         start = center - inherit_shape[i] // 2
    #         stop = start + inherit_shape[i]
    #         slices.append(slice(start, stop))
    #     curr_val[tuple(slices)] = subset_inherit
    #     return curr_val

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        # flops = 10e9
        for ii, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs, out_aux = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            if epoch==0 and ii==0:
                inputs = torch.randn(1, 3, 32, 32)
                inputs = Variable(inputs.cuda())
                params = sum(p.numel() for p in self.net.parameters())
                #flops1, params1 = profile(self.net, inputs=(inputs,))
                self.flops = profile_macs(copy.deepcopy(self.net), inputs)
                self.log_record('#parameters:%d, #FLOPs:%d' % (params, self.flops))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f'% (epoch+1, running_loss/total, (correct/total)))


    def final_train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        # flops = 10e9
        for ii, data in enumerate(self.full_trainloader, 0):
            inputs, labels = data
            # inputs = F.interpolate(inputs, size=40, mode='bicubic', align_corners=False)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs, out_aux = self.net(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            if epoch==0 and ii==0:
                inputs = torch.randn(1, 3, 32, 32)
                inputs = Variable(inputs.cuda())
                params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                #flops1, params1 = profile(self.net, inputs=(inputs,))
                self.flops = profile_macs(copy.deepcopy(self.net), inputs)
                self.log_record('#parameters:%d, #FLOPs:%d' % (params, self.flops))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f' % (epoch + 1, running_loss / total, (correct / total)))

    def get_flist(self, path):
        for root, dirs, files in os.walk(path):
            self.log_record(files)
            # pass
        return files

    def update_op(self, op_name, curr_acc, type=None):
        pool_path = './trained_models/'
        net_items = self.net.state_dict().items()
        updated_weights = {}

        for sub_key, val in net_items:
            if op_name in sub_key:
                updated_weights[sub_key] = val

        if type:
            updated_weights_path = pool_path + op_name + '-' + str(type) + '-' + str(curr_acc) + '.pt'
        else:
            updated_weights_path = pool_path+op_name+'-'+str(curr_acc)+'.pt'
        torch.save(updated_weights, updated_weights_path)

    def update_weight_pool(self, curr_acc):
        pool_path = './trained_models/'
        subparticle_length = Utils.get_params('PSO', 'particle_length') // 3
        subParticles = [self.particle[0:subparticle_length], self.particle[subparticle_length:2 * subparticle_length],
                        self.particle[2 * subparticle_length:]]
        weight_pool = self.get_flist(pool_path)
        curr_acc = curr_acc.cpu().numpy()
        curr_acc = np.around(curr_acc,4)
        for op_name in ['begin1', 'end1', 'linear', 'tranLayer0', 'tranLayer1']:
            is_found = False
            for op in weight_pool:
                if op_name in op:
                    is_found = True
                    op_acc = float(op.split('.pt')[0].split('-')[1])
                    if curr_acc > op_acc:
                        self.update_op(op_name, curr_acc)
                        os.remove(pool_path + op)
                        self.log_record('update new op: %s, and remove old op: %s'%(op_name+'-'+str(curr_acc), op))
            if not is_found:
                self.update_op(op_name, curr_acc)
                self.log_record('update new op: %s' % (op_name + '-' + str(curr_acc)))

        for j, subParticle in enumerate(subParticles):
            for number, dimen in enumerate(subParticle):
                type = int(dimen)
                if not type == 0:
                    is_found = False
                    op_name = 'evolved_block'+str(j)+'.layer.'+str(number)
                    for op in weight_pool:
                        if op_name+'-'+str(type)+'-' in op:
                            is_found = True
                            op_acc = float(op.split('.pt')[0].split('-')[-1])
                            if curr_acc > op_acc:
                                self.update_op(op_name, curr_acc, type)
                                os.remove(pool_path + op)
                                self.log_record('update new op: %s, and remove old op: %s' % (op_name + '-'+ str(type)+'-' + str(curr_acc), op))
                    if not is_found:
                        self.update_op(op_name, curr_acc, type)
                        self.log_record('update new op: %s' % (op_name + '-' + str(type) + '-' + str(curr_acc)))

    def validate(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        is_terminate = 0
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs, _ = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = correct / total
            self.update_weight_pool(self.best_acc)
        if epoch >= self.best_epoch + 3 or correct / total - self.best_acc < -0.03:
            is_terminate = 1
        self.log_record(
            'Validate-Epoch:%4d,  Validate-Loss:%.4f, Acc:%.4f' % (epoch + 1, test_loss / total, correct / total))
        return is_terminate

    def process(self):
        min_epoch_train = Utils.get_params('SEARCH', 'min_epoch_train')
        min_epoch_eval = Utils.get_params('SEARCH', 'min_epoch_eval')

        lr_rate = 0.03
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=5e-5, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, min_epoch_eval)

        is_terminate = 0
        params = sum(p.numel() for p in self.net.parameters())
        # self.validate(0)
        for p in range(min_epoch_train):
            if not is_terminate:
                self.train(p, optimizer)
                scheduler.step()
                is_terminate = self.validate(p)
            else:
                return self.best_acc, params, self.flops
        return self.best_acc, params, self.flops

    def process_test(self):
        params = sum(p.numel() for n,p in self.net.named_parameters() if p.requires_grad and not n.__contains__('auxiliary'))
        total_epoch = Utils.get_params('SEARCH', 'epoch_test')
        lr_rate = 0.0004
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

        # self.test(0)
        for p in range(total_epoch):
            if p < 5:
                optimizer_ini = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=self.weight_decay)
                self.final_train(p, optimizer_ini)
                self.test(p)
            else:
                self.final_train(p, optimizer)
                self.test(p)
                scheduler.step()
        return self.best_acc, params, self.flops

    def test(self, p):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0

        all_predictions = []
        all_labels = []

        for _, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs, _ = self.net(inputs)

            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            outputs = F.softmax(outputs, dim=1)
            all_predictions.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())

        roc_auc = roc_auc_score(all_labels, all_predictions, multi_class='ovo', average='weighted')
        cohen_kappa = cohen_kappa_score(all_labels, np.argmax(all_predictions, axis=1))
        f1 = f1_score(all_labels, np.argmax(all_predictions, axis=1), average='weighted')
        IBA = balanced_accuracy_score(all_labels, np.argmax(all_predictions, axis=1))

        if correct / total > self.best_acc:
            torch.save(self.net.state_dict(), './trained_models/best_CNN.pt')
            self.best_acc = correct / total

        self.log_record(
            'Test-Loss:%.4f, Acc:%.4f, cohen_kappa:%.4f, ROC-AUC:%.4f, F1-Score:%.4f, IBA:%.4f' % (
                test_loss / total, correct / total, cohen_kappa, roc_auc, f1, IBA))

class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, particle=None, batch_size=None, weight_decay=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        params = 1e9
        flops = 10e9
        try:
            m = TrainModel(is_test, particle, batch_size, weight_decay)
            m.log_record('Used GPU#%s, worker name:%s[%d]' % (gpu_id, multiprocessing.current_process().name, os.getpid()))
            if is_test:
                best_acc, params, flops = m.process_test()
            else:
                best_acc, params, flops = m.process()
            #import random
            #best_acc = random.random()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.4f'%best_acc)
            m.log_record('Finished-Err:%.4f' % (1-best_acc))

            f1 = open('./populations/err_%02d.txt'%(curr_gen), 'a+')
            f1.write('%s=%.5f\n'%(file_id, 1-best_acc))
            f1.flush()
            f1.close()

            f2 = open('./populations/params_%02d.txt' % (curr_gen), 'a+')
            f2.write('%s=%d\n' % (file_id, params))
            f2.flush()
            f2.close()

            f3 = open('./populations/flops_%02d.txt' % (curr_gen), 'a+')
            f3.write('%s=%d\n' % (file_id, flops))
            f3.flush()
            f3.close()
"""
