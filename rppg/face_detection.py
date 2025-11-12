import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch

'''
# 初始化 OpenFace 人脸对齐器
align = align_dlib.AlignDlib('shape_predictor_68_face_landmarks.dat')

# 设定裁剪后的图像大小
target_size = 128

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # 检测人脸
            bb = align.getLargestFaceBoundingBox(frame)
            if bb is not None:
                # 对齐人脸
                landmarks = align.findLandmarks(frame, bb)
                left = min(landmarks[:, 0])
                right = max(landmarks[:, 0])
                top = min(landmarks[:, 1])
                bottom = max(landmarks[:, 1])
                width = right - left
                height = bottom - top

                # 计算边界框大小
                bbox_width = int(width * 1.2)
                bbox_height = int(height * 1.2)

                # 计算中心面部点
                center_x = left + width // 2
                center_y = top + height // 2

                # 计算边界框的坐标
                x1 = center_x - bbox_width // 2
                y1 = center_y - bbox_height // 2
                x2 = x1 + bbox_width
                y2 = y1 + bbox_height

                # 确保边界框在图像内部
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # 裁剪人脸并调整大小
                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, (target_size, target_size))
                face = transform(face).unsqueeze(0)
                frames.append(face)
        else:
            break

    cap.release()
    return torch.cat(frames, dim=0)


# 调用示例
video_path = 'your_video.mp4'
preprocessed_video = preprocess_video(video_path)
print("预处理后的视频张量形状:", preprocessed_video.shape)

'''
def face_detection(video_path):

    device = torch.device('cuda')
    mtcnn = MTCNN(device=device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    N = 0
    video_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        N += 1

        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_list.append(frame)
        else:
            break

        # if N/fps>60: # only get the first 60s video
        #     break

    cap.release()

    face_list = []
    for t, frame in enumerate(video_list):
        if t==0:
            boxes, _, = mtcnn.detect(frame) # we only detect face bbox in the first frame, keep it in the following frames.
        if t==0:
            box_len = np.max([boxes[0,2]-boxes[0,0], boxes[0,3]-boxes[0,1]])
            box_half_len = np.round(box_len/2*1.1).astype('int')
        box_mid_y = np.round((boxes[0,3]+boxes[0,1])/2).astype('int')
        box_mid_x = np.round((boxes[0,2]+boxes[0,0])/2).astype('int')
        cropped_face = frame[box_mid_y-box_half_len:box_mid_y+box_half_len, box_mid_x-box_half_len:box_mid_x+box_half_len]
        cropped_face = cv2.resize(cropped_face, (128, 128))
        face_list.append(cropped_face)
        
        print('face detection %2d'%(100*(t+1)/len(video_list)), '%', end='\r', flush=True)

    face_list = np.array(face_list) # (T, H, W, C)
    face_list = np.transpose(face_list, (3,0,1,2)) # (C, T, H, W)
    face_list = np.array(face_list)[np.newaxis]

    return face_list, fps
