# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, pdb, sys, glob
import argparse
from os import path as osp

def generate_extract_300vw(P300VW):
  allfiles = glob.glob(osp.join(P300VW, '*'))
  alldirs = []
  for xfile in allfiles:
    if osp.isdir( xfile ):
      alldirs.append(xfile)
  assert len(alldirs) == 114, 'The directories of 300VW should be 114 not {}'.format(len(alldirs))
  cmds = []
  for xdir in alldirs:
    video = osp.join(xdir, 'vid.avi')
    exdir = osp.join(xdir, 'extraction')
    if not osp.isdir(exdir): os.makedirs(exdir)
    cmd = 'ffmpeg -i {:} {:}/%06d.png'.format(video, exdir)
    cmds.append( cmd )

  if not osp.isdir('./cache'):
    os.makedirs('./cache')

  with open('./cache/Extract300VW.sh', 'w') as txtfile:
    for cmd in cmds:
      txtfile.write('{}\n'.format(cmd))
  txtfile.close()

if __name__ == '__main__':
  paser = argparse.ArgumentParser('300vw')
  paser.add_argument('-d', '--dir', type=str, required=True, help='300vw dir')
  args = paser.parse_args()
  assert osp.isdir(args.dir), 'path {} does not exist!'.format(args.dir)
  generate_extract_300vw(args.dir)
