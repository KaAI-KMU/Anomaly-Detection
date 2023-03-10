# Purpose

Make pickle files that include Optical flow, BBox, Frame id, Ego motion

```bash
C:.
├─── data
|     └─ {video_name}.npy (Bbox)
|     └─ {video_name}.txt (Ego motion)
|     └─ {video_name}
|     └─ {Optical_flow_files}.flo
└─── result
    └─ {video_name}
            └─ {video_name}_{Object id}_{start_frame}.pkl
```
