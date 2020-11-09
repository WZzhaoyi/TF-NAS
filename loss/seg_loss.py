"""Custom losses."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
import torch.distributed as dist


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            if i == 3:
                # aux_loss = nn.L1Loss()
                # aux_loss = 0.00001 * aux_loss(preds[i], images)
                # loss += aux_loss
                continue
            else:
                aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
                loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        
        if self.aux:
            inputs = tuple(list(preds) + [target])
            return dict(loss=self._aux_forward(*inputs))
        # elif len(preds) > 1:
        #     return dict(loss=self._multiple_forward(*inputs))
        else:
            inputs = tuple([preds]+[target])
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses
