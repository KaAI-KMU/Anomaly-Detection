import cv2
import os

saving_path = 'D:/Dachcam_dataset/videos/training/frames/'
original_path = 'D:/Dachcam_dataset/videos/training/positive/'

def video_capture(video_path, saving_path):
    print(video_path)
    vidcap = cv2.VideoCapture(video_path)
    past_frame = 0
    while(vidcap.isOpened()):
        ret, image = vidcap.read() # image = 현재 이미지 -> 저장 대상
        frame = int(vidcap.get(1))
        if past_frame == frame: # 마지막이라면 종료
            break
        
        # 저장 될 위치 -> result/video_name/{frame_number}.jpg
        path = saving_path +str(frame-1).zfill(6)+'.jpg'
        # 이미지 저장
        cv2.imwrite(path,image)
        #print('Saved frame%d.jpg' % count)

        if frame % 500 == 0:
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            print(path)
        past_frame = frame


video_names = [instance.split('.')[0] for instance in os.listdir(original_path)]
for video_name in video_names:
    if not os.path.isdir(f'{saving_path}{video_name}'):
        os.mkdir(f'{saving_path}{video_name}')
    video_capture(f'{original_path}{video_name}.mp4', f'{saving_path}{video_name}/')