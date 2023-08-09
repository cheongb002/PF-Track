from typing import List, Tuple
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.registry import MODELS
import torch
from mmengine.utils import is_seq_of
import numpy as np
from typing import List, Tuple, Union

@MODELS.register_module()
class TrackDataPreprocessor(Det3DDataPreprocessor):
    def forward(self,
                data: List[dict],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict or List[dict]): Data from dataloader. The dict contains
                the whole batch data, when it is a list[dict], the list
                indicates test time augmentation.

                Expecting a data dict with a batch of data, where each
                sample in the batch is a clip of multiple frames
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict or List[dict]: Data in the same format as the model input.
        """
        # breakpoint()
        data_dict = {
            'data_samples': [],
            'inputs': {},
        }
        if not isinstance(data, list):
            data = [data]
        for batch_idx, data_i in enumerate(data):
            processed_data_i = self.simple_process(data_i, training)
            data_dict['data_samples'].append(processed_data_i['data_samples'])
            for key, val in processed_data_i['inputs'].items():
                if key not in data_dict['inputs']:
                    data_dict['inputs'][key] = []
                data_dict['inputs'][key].append(val)            
        # breakpoint()
        return data_dict
    
    def split_data_into_frames(self, data_i: dict) -> List[dict]:
        """Split data_i into multiple frames
        Args:
            data_i (dict): Data from dataloader. The dict contains
                the whole batch data, when it is a list[dict], the list
                indicates test time augmentation.
        Returns:
            List[dict]: List of data_i for each frame
        """
        data_queue = []
        for frame_idx in range(len(data_i['data_samples'])):
            data_queue.append(self.get_data_i_for_frame(data_i, frame_idx))
        return data_queue
    
    def get_data_i_for_frame(self, data_i: dict, frame_idx: int) -> dict:
        data_i_frame_j = {
            'data_samples': [data_i['data_samples'][frame_idx]],
            'inputs': {},
        }
        for key, val in data_i['inputs'].items():
            if is_seq_of(val, torch.Tensor):
                data_i_frame_j['inputs'][key] = [val[frame_idx]]
            else:
                data_i_frame_j['inputs'][key] = [val]
        return data_i_frame_j
        
