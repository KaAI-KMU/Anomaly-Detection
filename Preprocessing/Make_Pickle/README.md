# Purpose

Make pickle files that include Optical flow, BBox, Frame id, Ego motion

C:.
├─data
│ └─ {video\*name}.npy (Bbox)
| └─ {video\*name}.txt (Ego motion)
| └─ {video\*name}
| └─ {Optical*flow_files}.flo
└─result
└─ {video_name}
└─ {video_name}*{Object id}\_{start_frame}.pkl
