# Purpose

Make pickle files that include Optical flow, BBox, Frame id, Ego motion

```bash
C:.
├─data
│  └─662b0755-99d38907
│  └─make_pickle.py
└─result
    └─{video_name}
            └─{video_name}_{object_id}_{start_frame}.pkl

D:.
└─bdd100k_40
     ├─{video_name}.npy # BBox
     │
     ├─{video_name}.txt # Ego Motion
     │
     └─flow
        └─000001.flo
        └─000002.flo
        └─000003.flo
```

It will read videos data in D drive and make one pickle file in C drive

If you have different data path change line 6, 8, 36, 67, 76

If you want to change saving path, change line 64

'THRESHOLD' is value that endure can't detect object's bbox. Default value is 3.

'PASS_SIZE' minimum length that will be saved

If there is 2 frame that can't detect object bbox, it will interpolate the value that missed it by using linspace.
