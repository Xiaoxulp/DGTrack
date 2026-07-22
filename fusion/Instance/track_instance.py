import numpy as np
from fusion.Instance.basetrack import BaseTrack,TrackState

class STrack(BaseTrack):
    def __init__(self,track_id,bbox_tlxy,boxes_norm,feat=None): 
        super().__init__()
        self.track_id = track_id
        self.bbox_tlxy = bbox_tlxy
        self.boxes_norm = boxes_norm 
        self.is_activated = False
        self.tracklet_len = 0 
        self.state = TrackState.New
        self.reid_feature = None
        if feat is not None: 
            self.reid_feature = feat
    def __repr__(self):
        
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
