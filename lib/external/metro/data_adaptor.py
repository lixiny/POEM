import os

import cv2
import numpy as np
import torch
from lib.utils.builder import DATASET
from lib.utils.logger import logger
from termcolor import cprint
from lib.utils.config import CN


class METRODataConverter(object):

    def __init__(self, is_train) -> None:
        ###################################
        # Masking percantage
        # We observe that 0% or 5% works better for 3D hand mesh
        # We think this is probably becasue 3D vertices are quite sparse in the down-sampled hand mesh
        self.mvm_percent = 0.0  # or 0.05
        ###################################
        self.is_train = is_train

    def convert(self, inputs):
        mjm_mask = np.ones((21, 1), dtype=np.float32)
        if self.is_train:
            num_joints = 21
            pb = np.random.random_sample()
            masked_num = int(pb * self.mvm_percent * num_joints)  # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_joints), replace=False, size=masked_num)
            mjm_mask[indices, :] = 0.0

        mvm_mask = np.ones((195, 1), dtype=np.float32)
        if self.is_train:
            num_vertices = 195
            pb = np.random.random_sample()
            masked_num = int(pb * self.mvm_percent * num_vertices)  # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices), replace=False, size=masked_num)
            mvm_mask[indices, :] = 0.0

        inputs["target_verts_3d"] = inputs["target_verts_3d"][:778]
        inputs["target_verts_uvd"] = inputs["target_verts_uvd"][:778]
        inputs["target_verts_3d_rel"] = inputs["target_verts_3d_rel"][:778]

        inputs.update({"mjm_mask": mjm_mask, "mvm_mask": mvm_mask})
        return inputs
