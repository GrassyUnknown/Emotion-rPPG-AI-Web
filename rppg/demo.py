import io
import torch
import cv2, dlib, time, math
import numpy as np
import matplotlib.pyplot as plt
from .PhysNetModel import PhysNet
from .ResNet3D import generate_model
from .utils_sig import butter_bandpass, hr_fft
from .face_detection import face_detection, face_detection_align
from .roi_face_det import crop_face

def cal_psd_rr(output, Fs, return_type):
    """
    基于rPPG信号的功率谱密度估计呼吸率（Respiratory Rate, RR）
    呼吸频率范围：0.1~0.5Hz → 6~30 次/分钟
    """
    cur_device = output.device
    
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = 2 * math.pi * torch.arange(0, N, dtype=torch.float) / N
        hanning = torch.from_numpy(np.hanning(N)).type(torch.FloatTensor).view(1, -1)

        k = k.type(torch.FloatTensor).to(cur_device)
        two_pi_n_over_N = two_pi_n_over_N.to(cur_device)
        hanning = hanning.to(cur_device)
        
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute
    
    output = output.view(1, -1)
    N = output.size()[1]
    
    # 呼吸率范围：6~30 次/分钟 → 转换为Hz（0.1~0.5Hz）
    rr_range = torch.arange(6, 30, dtype=torch.float)
    unit_per_hz = Fs / N
    feasible_rr_hz = rr_range / 60.0  # 转换为Hz
    k = feasible_rr_hz / unit_per_hz
    
    complex_absolute = compute_complex_absolute_given_k(output, k, N)
    complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute
    complex_absolute = complex_absolute.view(-1)
    whole_max_val, whole_max_idx = complex_absolute.max(0)
    whole_max_idx = whole_max_idx.type(torch.float)

    if return_type == 'psd':
        return complex_absolute
    elif return_type == 'rr':
        return whole_max_idx + 6  # 对应6~30次/分钟的起始值

def cal_psd_hr(output, Fs, return_type):
    
    cur_device = output.device
    
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = 2 * math.pi * torch.arange(0, N, dtype=torch.float) / N
        hanning = torch.from_numpy(np.hanning(N)).type(torch.FloatTensor).view(1, -1)

        k = k.type(torch.FloatTensor).to(cur_device)
        two_pi_n_over_N = two_pi_n_over_N.to(cur_device)
        hanning = hanning.to(cur_device)
        
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute
    
    output = output.view(1, -1)
    
    N = output.size()[1]
    bpm_range = torch.arange(40, 180, dtype=torch.float)
    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz
    
    # only calculate feasible PSD range [0.7,4]Hz
    complex_absolute = compute_complex_absolute_given_k(output, k, N)
    complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute
    complex_absolute = complex_absolute.view(-1)
    whole_max_val, whole_max_idx = complex_absolute.max(0) # max返回（values, indices）
    whole_max_idx = whole_max_idx.type(torch.float) # 功率谱密度的峰值对应频率即为心率

    if return_type == 'psd':
        return complex_absolute
    elif return_type == 'hr':
        return whole_max_idx + 40	# Analogous Softmax operator

def analyze_heart_rate(video_path: str, gpu_id=0):
    """
    输入：
        video_path (str): 视频文件路径
    输出：
        hr (float): 估计的心率 (bpm)
        rr (float): 估计的呼吸率 (次/分钟)
        image_bytes (bytes): matplotlib 生成的图像字节流，可用于 Streamlit 显示
    """
    # ----------------------------
    # 1. 人脸检测并提取视频帧（保留原逻辑）
    # ----------------------------
    face_list, fps = face_detection(video_path=video_path)
    
    # ----------------------------
    # 2. 模型推理（保留原逻辑）
    # ----------------------------
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        face_list = torch.tensor(face_list.astype('float32')).to(device)
        rppg_model = PhysNet(S=2).to(device).eval()
        rppg_model.load_state_dict(torch.load('/home/zhangzijie/Emotion-rPPG-AI-Web/rppg/model_weights.pt', map_location=device))
        rppg_model_signal = rppg_model(face_list * 255)[:, -1, :]

        model = generate_model(model_depth=18, num_frames=face_list.shape[2]).to(device).eval()
        model.load_state_dict(torch.load('/home/zhangzijie/Emotion-rPPG-AI-Web/rppg/rppg_estimator_stu_dataset_0.pth', map_location=device))

        inputs = {'input_clip': face_list}
        rppg_gpu = model(inputs)['rPPG']
        rppg = rppg_gpu[0].detach().cpu().numpy()
        rppg = butter_bandpass(rppg, lowcut=0.6, highcut=3, fs=fps)

        # ----------------------------
        # 新增：心率/呼吸率分片段估计（时序序列）
        # ----------------------------
        clip_len = 90
        frame_len = rppg.shape[0]
        num_clip = frame_len // clip_len
        
        hr_long_period = []
        rr_long_period = []
        for i in range(num_clip):
            # 心率估计
            hr_clip = cal_psd_hr(rppg_gpu[:, i * clip_len: (i + 1) * clip_len], fps, return_type='hr')
            # 呼吸率估计
            rr_clip = cal_psd_rr(rppg_gpu[:, i * clip_len: (i + 1) * clip_len], fps, return_type='rr')
            
            hr_long_period.extend([hr_clip.cpu().data.numpy()] * clip_len)
            rr_long_period.extend([rr_clip.cpu().data.numpy()] * clip_len)

    # ----------------------------
    # 3. 计算最终HR/RR + 绘图（核心修改）
    # ----------------------------
    # 最终心率/呼吸率（取功率谱峰值）
    psd_x_hr = cal_psd_hr(rppg_gpu, fps, return_type='psd')
    hr = psd_x_hr.view(-1).max(0)[1].cpu() + 40
    
    psd_x_rr = cal_psd_rr(rppg_gpu, fps, return_type='psd')
    rr = psd_x_rr.view(-1).max(0)[1].cpu() + 6

    # 绘图：第一行2个子图，第二行1个双轴大子图
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1])

    # 子图1：rPPG波形
    ax1 = fig.add_subplot(gs[0, 0])
    rppg_model_signal_np = rppg_model_signal.view(-1).cpu().numpy()
    ax1.plot(rppg_model_signal_np, color='blue', linewidth=2, label='rPPG Signal')
    ax1.set_ylabel('Amplitude', fontsize=25)
    ax1.tick_params(labelsize=20)
    ax1.legend(fontsize=20, loc='upper right')
    ax1.set_title('rPPG Waveform', fontsize=25)

    # 子图2：HR/ RR 功率谱密度
    ax2 = fig.add_subplot(gs[0, 1])
    # 心率PSD
    bpm_range = np.arange(40, 180)
    ax2.plot(bpm_range, psd_x_hr.cpu().numpy(), color='red', linewidth=2, label='HR PSD')
    # 标注峰值
    ax2.axvline(x=hr, color='red', linestyle='--', linewidth=2, label=f'HR={hr:.1f} bpm')
    # 图例合并
    lines1, labels1 = ax2.get_legend_handles_labels()
    ax2.legend(lines1, labels1, fontsize=18, loc='upper right')
    ax2.set_xlabel('Rate', fontsize=25)
    ax2.set_ylabel('HR PSD', fontsize=25, color='red')
    ax2.tick_params(labelsize=20)
    ax2.set_title('Power Spectral Density', fontsize=25)

    # 子图3：HR/RR时序变化（双轴+左右标注数值）
    ax3 = fig.add_subplot(gs[1, :])
    # 左y轴：心率
    ax3.plot(hr_long_period, color='red', linewidth=2, label=f'Heart Rate (HR)')
    ax3.set_ylabel('Heart Rate (bpm)', fontsize=25, color='red')
    ax3.tick_params(axis='y', labelsize=20, colors='red')
    ax3.set_ylim([40, 180])
    # 右y轴：呼吸率
    ax3_twin = ax3.twinx()
    ax3_twin.plot(rr_long_period, color='green', linewidth=2, label=f'Respiratory Rate (RR)')
    ax3_twin.set_ylabel('Respiratory Rate (rpm)', fontsize=25, color='green')
    ax3_twin.tick_params(axis='y', labelsize=20, colors='green')
    ax3_twin.set_ylim([0, 20])
    # 全局设置
    ax3.set_xlabel('Frame', fontsize=25)
    ax3.set_title('Heart Rate & Respiratory Rate Over Time', fontsize=25)
    ax3.tick_params(axis='x', labelsize=20)
    # 左右标注最终数值（固定位置）
    ax3.text(0.02, 0.95, f'HR: {hr:.1f} bpm', transform=ax3.transAxes, 
             fontsize=22, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax3_twin.text(0.98, 0.95, f'RR: {rr:.1f} rpm', transform=ax3.transAxes, 
                  fontsize=22, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    # 合并图例
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3_twin, labels3_twin = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines3_twin, labels3 + labels3_twin, fontsize=20, loc='lower right')

    # 调整布局
    plt.tight_layout()

    # 保存图像到内存
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.read()

    # ----------------------------
    # 4. 返回结果（新增rr）
    # ----------------------------
    return hr, image_bytes


if __name__ == "__main__":
    video_path = '/data/wujunjie/cohface/6/1/data.avi'  # 替换为你的视频文件路径
    hr, image_bytes = analyze_heart_rate(video_path, gpu_id=0)
    print(f"Estimated Heart Rate: {hr} bpm")
    # 你可以将 image_bytes 保存为文件或在 Streamlit 中显示