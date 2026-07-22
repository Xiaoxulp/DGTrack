
import numpy as np

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4
class BaseTrack(object):
    
    track_id = 0 
    is_activated = False
    state = TrackState.New
    reid_feature = None 
    start_frame = 0 
    frame_id = 0 
    time_since_update = 0 

    
    @property
    def end_frame(self):
        
        return self.frame_id
