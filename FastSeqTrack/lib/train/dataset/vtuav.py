import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class VTUAV(BaseVideoDataset):
    """ LasHeR dataset(aligned version).

    Publication:
        A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/pdf/2104.13202.pdf

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None,
                 multi_modal_vision=False, multi_modal_language=False, use_nlp=False):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().vtuav_dir if root is None else root
        assert split in ['train', 'val','all'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('VTUAV', root)
        self.dtype = dtype

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        self.init_idx = np.load(os.path.abspath(os.path.join(self.root, 'init_frame.npy')),
                                allow_pickle=True).item()

        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language
        self.use_nlp = use_nlp

    def get_name(self):
        return 'vtuav'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _get_sequence_list(self):
        seq_dir = os.path.join(self.root, 'trainval')
        return [seq for seq in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, seq))]

    def _read_bb_anno(self, seq_path):
        # in lasher dataset, visible.txt is same as infrared.txt
        rgb_bb_anno_file = os.path.join(seq_path, "rgb.txt")
        # ir_bb_anno_file = os.path.join(seq_path, "infrared.txt")
        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=' ', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        # ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(rgb_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, "trainval", self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        """2022/8/10 ir and rgb have synchronous w=h=0 frame_index"""
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # Note original filename is chaotic, we rename them
        seq_name = seq_path.split('/')[-1]
        if seq_name in self.init_idx:
            init_idx = self.init_idx[seq_name]
        else:
            init_idx = 0
        rgb_frame_path = os.path.join(seq_path, 'rgb', '{:06d}.jpg'.format(frame_id*10+init_idx))  # frames start from 0
        ir_frame_path = os.path.join(seq_path, 'ir', '{:06d}.jpg'.format(frame_id*10+init_idx))
        return (rgb_frame_path, ir_frame_path)  # jpg jpg

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img  # (h,w,6)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
