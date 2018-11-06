# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, os
from os import path as osp
import numpy as np
import cv2
import torch
from copy import deepcopy
from pathlib import Path
from xvision import Eval_Meta
from log_utils import AverageMeter, time_for_file, convert_secs2time
from .losses import compute_regression_loss
import torchvision
import visdom
vis = None


def plt_landmark(image, landmark, line_width=2, color=(255, 255, 255)):
    image_copy = image.copy()
    for x, y in landmark:
        ix, iy = map(int, [x, y])
        cv2.circle(image_copy, (ix, iy), line_width, color, line_width)
    return image_copy


def vis_plt(batch_image, points):
    mean = batch_image.new_tensor([0.485, 0.456, 0.406]).view(-1, 3, 1, 1)
    std = batch_image.new_tensor([0.229, 0.224, 0.225]).view(-1, 3, 1, 1)
    images = batch_image * std + mean
    images *= 255
    images = images.numpy().transpose(0, 2, 3, 1).astype(np.int8)
    landmark = points.numpy().astype(np.int32)
    for i in range(images.shape[0]):
        images[i] = plt_landmark(images[i], landmark[i], line_width=1, color=(255, 0, 0))
    grid_images = torchvision.utils.make_grid(torch.from_numpy(images.transpose(0, 3, 1, 2).astype(np.float32)), nrow=8)
    return grid_images


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

  global vis
  if vis is None:
      vis = visdom.Visdom(env='{}'.format(osp.basename(args.model_config)))

  end = time.time()
  for i, (inputs, mask, points, image_index, nopoints, cropped_size) in enumerate(loader):
    # inputs : Batch, Channel, Height, Width
    image_index = image_index.numpy().squeeze(1).tolist()
    batch_size, num_pts = inputs.size(0), args.num_pts
    visible_point_num   = float(np.sum(mask.numpy()[:,:-1])) / batch_size
    visible_points.update(visible_point_num, batch_size)
    nopoints    = nopoints.numpy().squeeze(1).tolist()
    annotated_num = batch_size - sum(nopoints)
    
    points = points[:, :, :2].contiguous()
    if i % 100 == 0:
        vis.image(vis_plt(inputs.cpu(), points), win='gt')
    # measure data loading time
    points = points.cuda(non_blocking=True)
    mask = mask.cuda(non_blocking=True)
    data_time.update(time.time() - end)
    #  print('points.size(): ', points.size())
    #  print('mask.size(): ', mask.size())

    # batch_heatmaps is a list for stage-predictions, each element should be [Batch, C, H, W]
    batch_predicts = net(inputs)
    forward_time.update(time.time() - end)
    if i % 100 == 0:
        vis.image(vis_plt(inputs.cpu(), batch_predicts.detach().cpu()), win='predict')
    
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
