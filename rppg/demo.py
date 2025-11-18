import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from .PhysNetModel import PhysNet
from .utils_sig import butter_bandpass, hr_fft
from .face_detection import face_detection


def analyze_heart_rate(video_path: str, gpu_id=0):
    """
    输入：
        video_path (str): 视频文件路径
    输出：
        hr (float): 估计的心率 (bpm)
        image_bytes (bytes): matplotlib 生成的图像字节流，可用于 Streamlit 显示
    """
    # ----------------------------
    # 1. 人脸检测并提取视频帧
    # ----------------------------
    face_list, fps = face_detection(video_path=video_path)

    # ----------------------------
    # 2. 模型推理
    # ----------------------------
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        face_list = torch.tensor(face_list.astype('float32')).to(device)
        model = PhysNet(S=2).to(device).eval()
        model.load_state_dict(torch.load('/home/zhangzijie/Emotion-rPPG-AI-Web/rppg/model_weights.pt', map_location=device))
        rppg = model(face_list)[:, -1, :]
        rppg = rppg[0].detach().cpu().numpy()
        rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)

    # ----------------------------
    # 3. 计算心率并绘图
    # ----------------------------
    hr, psd_y, psd_x = hr_fft(rppg, fs=fps)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))
    ax1.plot(np.arange(len(rppg)) / fps, rppg)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('rPPG waveform')
    ax1.grid(True)

    ax2.plot(psd_x, psd_y)
    ax2.set_xlabel('Heart rate (bpm)')
    ax2.set_xlim([40, 200])
    ax2.set_ylabel('Power')
    ax2.set_title('PSD')
    ax2.grid(True)

    # 保存图像到内存
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.read()

    # ----------------------------
    # 4. 返回结果
    # ----------------------------
    return hr, image_bytes