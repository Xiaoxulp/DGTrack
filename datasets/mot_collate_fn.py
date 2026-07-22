from typing import List
import torch
def mot_collate_fn(batch: List[dict]) -> dict:
    '''
    dataset_item = {
        "imgs": images,
        "gt_instances": targets
    }
    :param batch: List[dataset_item]
    :return:
    '''
    ret_dict = {}
    for key in batch[0].keys():
        assert not isinstance(batch[0][key], torch.Tensor)
        ret_dict[key] = [sample[key] for sample in batch]
        if len(ret_dict[key]) == 1:
            ret_dict[key] = ret_dict[key][0]
    return ret_dict
