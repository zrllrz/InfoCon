import os
import cv2
from PIL import Image

file_dir = '/home/rzliu/AutoCoT/CoTPC-main/data/PegInsertionSide-v0/frames/0'
frames = list()

for i in range(138):
    frames.append(file_dir + f'/{i}.jpg')

print(frames)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
im0 = Image.open(file_dir + f'/0.jpg')
video = cv2.VideoWriter('vis_key_states/PegInsertionSide-v0/0/0.avi', fourcc, 60, im0.size)

for f in frames:
    frame = cv2.imread(f)
    print(frame)
    video.write(frame)

video.release()
