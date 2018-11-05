# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, os
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from xvision import Eval_Meta
from log_utils import AverageMeter, time_for_file, convert_secs2time
from .losses import compute_regression_loss

# train function (forward, backward, update)
def basic_train_regression(args, loader, net, criterion, optimizer, epoch_str, logger, opt_config):
  args = deepcopy(args)
  batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  visible_points, losses = AverageMeter(), AverageMeter()
  eval_meta = Eval_Meta()
  cpu = torch.device('cpu')

  # switch to train mode
  net.train()
  criterion.train()

  end = time.time()
  for i, (inputs, mask, points, image_index, nopoints, cropped_size) in enumerate(loader):
    # inputs : Batch, Channel, Height, Width
    image_index = image_index.numpy().squeeze(1).tolist()
    batch_size, num_pts = inputs.size(0), args.num_pts
    visible_point_num   = float(np.sum(mask.numpy()[:,:-1,:])) / batch_size
    visible_points.update(visible_point_num, batch_size)
    nopoints    = nopoints.numpy().squeeze(1).tolist()
    annotated_num = batch_size - sum(nopoints)
    
    points = points[:, :, :2].contiguous()

    # measure data loading time
    points = points.cuda(non_blocking=True)
    mask = mask.cuda(non_blocking=True)
    data_time.update(time.time() - end)

    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, C, H, W]
    batch_predicts = net(inputs)
    forward_time.update(time.time() - end)
    
    loss = compute_regression_loss(criterion, points, batch_predicts, mask)

    # measure accuracy and record loss
    losses.update(loss.item(), batch_size)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    eval_time.update(time.time() - end)

    np_batch_locs = batch_predicts.detach().to(cpu).numpy()
    cropped_size = cropped_size.numpy()
    # evaluate the training data
    for ibatch, (imgidx, nopoint) in enumerate(zip(image_index, nopoints)):
      if nopoint == 1: continue
      locations = np_batch_locs[ibatch,:,:]
      scores = np.ones((locations.shape[0],1), dtype=locations.dtype)
      xpoints = loader.dataset.labels[imgidx].get_points()
      assert cropped_size[ibatch,0] > 0 and cropped_size[ibatch,1] > 0, 'The ibatch={:}, imgidx={:} is not right.'.format(ibatch, imgidx, cropped_size[ibatch])
      scale_h, scale_w = cropped_size[ibatch,0] * 1. / inputs.size(-2) , cropped_size[ibatch,1] * 1. / inputs.size(-1)
      locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[ibatch,2], locations[:, 1] * scale_h + cropped_size[ibatch,3]
      assert xpoints.shape[1] == num_pts and locations.shape[0] == num_pts and scores.shape[0] == num_pts, 'The number of points is {} vs {} vs {} vs {}'.format(num_pts, xpoints.shape, locations.shape, scores.shape)
      # recover the original resolution
      prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)
      image_path = loader.dataset.datas[imgidx]
      face_size  = loader.dataset.face_sizes[imgidx]
      eval_meta.append(prediction, xpoints, image_path, face_size)

    # measure elapsed time
    batch_time.update(time.time() - end)
    last_time = convert_secs2time(batch_time.avg * (len(loader)-i-1), True)
    end = time.time()

    if i % args.print_freq == 0 or i+1 == len(loader):
      logger.log(' -->>[Train]: [{:}][{:03d}/{:03d}] '
                'Time {batch_time.val:4.2f} ({batch_time.avg:4.2f}) '
                'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '
                'Forward {forward_time.val:4.2f} ({forward_time.avg:4.2f}) '
                'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
                    epoch_str, i, len(loader), batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, loss=losses)
                  + last_time \
                  + ' In={:}, Out={:}'.format(list(inputs.size()), list(batch_predicts.size())) \
                  + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg))
  nme, _, _ = eval_meta.compute_mse(logger)
  return losses.avg, nme
