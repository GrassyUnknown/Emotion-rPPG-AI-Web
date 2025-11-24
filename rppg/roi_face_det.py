import numpy as np
import cv2

def get_roi_n_border_by_dlib(img, detector, predictor):

    def get_n_split_point(p1, p2, n):
        # n 等分 p1, p2 组成的线段
        x1, y1 = p1
        x2, y2 = p2
        x_np = np.linspace(x1, x2, n+1)
        y_np = np.linspace(y1, y2, n+1)
        return np.stack([x_np, y_np], axis=1).astype(np.int32)
    
    # get landmarks
    landmarks_first = []
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmarks_first.append(np.array([[p.x, p.y] for p in shape.parts()]))
    # print(len(landmarks_first), landmarks_first[0].shape)
    landmarks_first = np.array(landmarks_first)[0]
    # print(f"landmarks_first: {landmarks_first.shape}")
    actual_border = []
    # 获取脸颊部分的ROI
    border_cheek = []
    eye_center = np.mean(landmarks_first[36:48], axis=0).astype(np.int32)
    mouth_center = np.mean(landmarks_first[48:55], axis=0).astype(np.int32)
    cheek_left = [
        landmarks_first[1], landmarks_first[2], landmarks_first[3], landmarks_first[4]
    ]
    cheek_right = [
        landmarks_first[15], landmarks_first[14], landmarks_first[13], landmarks_first[12]
    ]
    cheek_center = get_n_split_point(eye_center, mouth_center, 3)
    for i in range(3):
        for j in range(3):
            border = np.concatenate((get_n_split_point(cheek_left[i], cheek_center[i], 3)[j:j+2],
                    get_n_split_point(cheek_left[i+1], cheek_center[i+1], 3)[j:j+2][::-1]), axis=0)
            border_cheek.append(border)
            border = np.concatenate((get_n_split_point(cheek_center[i], cheek_right[i], 3)[j:j+2],
                    get_n_split_point(cheek_center[i+1], cheek_right[i+1], 3)[j:j+2][::-1]), axis=0)
            border_cheek.append(border)
            # if i == 1 and (j == 1 or j == 2):
            #     actual_border.append(border)
    left_cheek_mean = np.mean(np.array(cheek_left[1:]), axis=0)
    center_cheek_mean = np.mean(np.array(cheek_center), axis=0)
    right_cheek_mean = np.mean(np.array(cheek_right[1:]), axis=0)
    left_cheek_center = ((4*left_cheek_mean[0] + 2*center_cheek_mean[0])/3, left_cheek_mean[1] + center_cheek_mean[1])
    right_cheek_center = ((4*right_cheek_mean[0] + 2*center_cheek_mean[0])/3, right_cheek_mean[1] + center_cheek_mean[1])

    # 获取下巴部分的ROI
    border_chin = []
    chin_down = [
        landmarks_first[4], landmarks_first[6], landmarks_first[8], landmarks_first[10], landmarks_first[12]
    ]
    chin_up = [
        landmarks_first[59], landmarks_first[58], landmarks_first[57], landmarks_first[56], landmarks_first[55]
    ]
    for i in range(4):
        border_chin.append(np.array([chin_up[i], chin_up[i+1], chin_down[i+1], chin_down[i]]))

    # 获取额头部分的ROI
    border_forehead = []
    fore_head_left = get_n_split_point(landmarks_first[68], landmarks_first[77], 2)
    fore_head_right = get_n_split_point(landmarks_first[73], landmarks_first[78], 2)
    for i in range(2):
        for j in range(5):
            border = np.concatenate((get_n_split_point(fore_head_left[i], fore_head_right[i], 5)[j:j+2],
                    get_n_split_point(fore_head_left[i+1], fore_head_right[i+1], 5)[j:j+2][::-1]), axis=0)
            border_forehead.append(border)
            actual_border.append(border)

    roi_n_border = border_cheek + border_chin  + border_forehead
    forehead_center = ((landmarks_first[74][0] + landmarks_first[75][0]), (landmarks_first[74][1] + landmarks_first[75][1]))
    return np.array(roi_n_border).astype(np.int32), forehead_center, left_cheek_center, right_cheek_center

def get_roi_n_border(img, n, mode, detector, predictor):
    # img (H, W, C)
    # n roi 的数量
    # mode roi 的划分方式
    # 返回多个 roi 的边界，shape (n, 4, 2)
    h, w = img.shape[:2]
    roi_size = int(np.sqrt(n))
    try:
        assert roi_size * roi_size == n
    except:
        raise ValueError('n must be square number')
    if mode == 'avg':
        roi_n_border = np.array([[(h // roi_size * i, w // roi_size * j), 
                            (h // roi_size * (i + 1) - 1, w // roi_size * j), 
                            (h // roi_size * (i + 1) - 1, w // roi_size * (j + 1) - 1), 
                            (h // roi_size * i, w // roi_size * (j + 1) - 1)] 
                            for i, j in [(m, n) for m in range(0,roi_size) for n in range(0,roi_size)]])
    elif mode == 'dlib':
        roi_n_border = get_roi_n_border_by_dlib(img.copy(), detector, predictor)
    return roi_n_border

def crop_face(last_roi, frame, detector, predictor):
    try:
        if last_roi is None:
            last_roi = {
                'forehead':[0, 0, 50, 20],
                'left_cheek':[0, 0, 20, 20],
                'right_cheek':[0, 0, 20, 20]
            }
        frame = frame.copy()
        det_frame = cv2.resize(frame, (320,240))
        roi_n_border, forehead_center, left_cheek_center, right_cheek_center = get_roi_n_border(det_frame, 25, 'dlib', detector, predictor)
        roi_n_border = roi_n_border * 2
        
        cropped_face_dict = {}
        roi_dict = {}
        for area, roi in last_roi.items():
            last_middle_point = ((last_roi[area][0]+last_roi[area][2])/2, (last_roi[area][1]+last_roi[area][3])/2)
            # print(middle_point)
            if area == 'forehead':
                middle_point = forehead_center
                rh, rw = 25, 10
            elif area == 'left_cheek':
                middle_point = left_cheek_center
                rh, rw = 10, 10
            elif area == 'right_cheek':
                middle_point = right_cheek_center
                rh, rw = 10, 10
            if np.abs(last_middle_point[0] - middle_point[0]) > 25 or np.abs(last_middle_point[1] - middle_point[1]) > 15:
                roi_dict[area] = [
                    int(middle_point[0] - rh),
                    int(middle_point[1] - rw),
                    int(middle_point[0] + rh),
                    int(middle_point[1] + rw)
                ]
                print(f'roi update : old middle : {last_middle_point}, current : {middle_point}')
            else:
                roi_dict[area] = last_roi[area]
            cropped_face_dict[area] = frame[roi_dict[area][1]:roi_dict[area][3], roi_dict[area][0]:roi_dict[area][2]]
        
        return cropped_face_dict, roi_n_border, roi_dict
    except Exception as e:
        print(e)
        return None, None, None
    # for i in range(roi_n_border.shape[0]):
    #     cv2.polylines(frame, [roi_n_border[i]], True, (255, 0, 0), 1)