import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from aiortc.contrib.media import MediaRecorder
import av
import cv2
import tempfile
import numpy as np
import time
import torch
import random
import sys
import os
from moviepy import VideoFileClip
import re
import whisper
import opencc

# ------------------------------
# AffectGPT 推理依赖
# ------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "AffectGPT"))
from affectgpt_inference import AffectGPTInference

# ------------------------------
# RPPG 推理依赖
# ------------------------------
from rppg.demo import analyze_heart_rate

gpu_id = 7

# =======================================
# 工具函数：音频提取
# =======================================
def extract_audio_from_video(video_path):
    """从视频中提取音频为 WAV 文件"""
    try:
        video_clip = VideoFileClip(video_path)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        video_clip.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
        video_clip.close()
        return temp_audio.name
    except Exception as e:
        st.error(f"音频提取失败：{e}")
        return None


# =======================================
# Streamlit 页面逻辑
# =======================================
st.set_page_config(page_title="情感识别与心率检测", layout="wide")
st.title("🎥 情感识别与心率检测 Demo")

# 持久缓存模型
@st.cache_resource(show_spinner=True)
def load_affectgpt_model():
    model = AffectGPTInference(
        cfg_path="/home/zhangzijie/web/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml",
        ckpt_path="/home/zhangzijie/ResearchFace/models/AffectGPT/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth",
        zeroshot=True,
        gpu_id=gpu_id
    )
    return model

try:
    model = load_affectgpt_model()
    st.success("✅ AffectGPT 模型已加载")
except Exception as e:
    st.error(f"⚠️ 模型加载失败：{e}")
    model = None

def display_emotion_result(result_text: str):

    # 1️⃣ 提取情感关键词部分
    match = re.search(r"emotional state is ([^.]+)[.]", result_text, re.IGNORECASE)
    emotions = []

    if match:
        emotion_part = match.group(1)
        # 统一分隔符
        emotion_part = re.sub(r"\band\b|/|&|;", ",", emotion_part)
        # 拆分情感词
        emotions = [e.strip(" ,") for e in emotion_part.split(",") if len(e.strip()) > 0]
        # 去重 & 首字母小写
        emotions = list(dict.fromkeys([e.lower() for e in emotions]))

    # 3️⃣ 输出展示
    st.markdown("### 🧠 检测到的情感状态")

    if emotions:
        colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#6a4c93"]
        emotion_tags = " ".join(
            [
                f"<span style='font-size:26px; font-weight:700; color:{colors[i % len(colors)]}; margin-right:10px;'>{e}</span>"
                for i, e in enumerate(emotions)
            ]
        )
        st.markdown(f"<div style='margin:10px 0;'>{emotion_tags}</div>", unsafe_allow_html=True)
    else:
        st.markdown("_未识别到明显的情感关键词。_")


st.markdown("""
该应用支持：
- 上传或录制视频；
- 自动提取视频音频；
- 使用 AffectGPT 模型分析情感状态；
- 使用 Contrast-Phys 分析心率状态；
""")

# 上传或拍摄视频
option = st.radio("选择输入方式：", ["上传视频文件", "使用摄像头拍摄"])
if option == "上传视频文件":
    uploaded_file = st.file_uploader("请上传视频文件（mp4 / mov / avi）", type=["mp4", "mov", "avi"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        st.session_state.video_path = temp_file.name
        st.video(st.session_state.video_path)
# 使用摄像头拍摄视频
elif option == "使用摄像头拍摄":
    def recorder_factory() -> MediaRecorder:
        return MediaRecorder('/tmp/record.mp4' , format="mp4")

    # 启动 WebRTC
    webrtc_streamer(
        key="record_only",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": True},  # 启用音视频
        in_recorder_factory=recorder_factory,
    )
    try:
        print(st.session_state.video)
    except AttributeError as e:
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        st.session_state.video = temp_file
        print(st.session_state.video)
    if st.button("完成录制"):
        with open('/tmp/record.mp4', "rb") as record:
            st.session_state.video.seek(0)
            st.session_state.video.truncate()
            st.session_state.video.write(record.read())
        st.session_state.video_path=st.session_state.video.name
        st.video(st.session_state.video_path)


# 字幕输入
st.subheader("💬 输入视频字幕或语音识别文字")
subtitle_text = st.text_area("请输入视频对应的文字内容（可选）", placeholder="若不输入，将自动识别音频。视频过长可能导致识别效果不佳。", height=100)


try:
    if st.button("分析情绪关键词"):
        # -----------------------------
        # Step 1: 提取音频
        # -----------------------------
        with st.spinner("正在提取音频..."):
            audio_path = extract_audio_from_video(st.session_state.video_path)
            if audio_path:
                st.success("✅ 音频提取成功")
                st.audio(audio_path)
        # -----------------------------
        # Step 2: 提取文本
        # -----------------------------
        if subtitle_text == '':
            with st.spinner("正在识别音频..."):
                whisper_model = whisper.load_model("small", f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
                result = whisper_model.transcribe(audio_path, initial_prompt="接下来是一段视频的字幕。Here are subtitles of a video.")
                print("Result of whisper:" + result['text'])
                subtitle_text = result['text']
                # converter = opencc.OpenCC("t2s.json")
                # subtitle_text = converter.convert(subtitle_text)
                st.success("✅ 音频识别成功，结果为：" + subtitle_text)
        # -----------------------------
        # Step 3: 情绪识别
        # -----------------------------
        with st.spinner("正在进行情绪识别..."):
            try:
                if model:
                    result = model.infer_emotion_ov(
                        video_path=st.session_state.video_path,
                        audio_path=audio_path,
                        subtitle=subtitle_text
                    )
                    st.success("✅ 情绪识别完成")
                    st.subheader("情绪识别结果：")
                    display_emotion_result(result)
                else:
                    st.error("模型加载失败，请稍后重新加载网页")

            except Exception as e:
                st.error(f"情绪识别出错：{e}")

    if st.button("描述情绪"):
        # -----------------------------
        # Step 1: 提取音频
        # -----------------------------
        with st.spinner("正在提取音频..."):
            audio_path = extract_audio_from_video(st.session_state.video_path)
            if audio_path:
                st.success("✅ 音频提取成功")
                st.audio(audio_path)
        # -----------------------------
        # Step 2: 提取文本
        # -----------------------------
        if subtitle_text == '':
            with st.spinner("正在识别音频..."):
                whisper_model = whisper.load_model("small", f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
                result = whisper_model.transcribe(audio_path, initial_prompt="接下来是一段视频的字幕。Here are subtitles of a video.")
                print("Result of whisper:" + result['text'])
                subtitle_text = result['text']
                # converter = opencc.OpenCC("t2s.json")
                # subtitle_text = converter.convert(subtitle_text)
                st.success("✅ 音频识别成功，结果为：" + subtitle_text)
        # -----------------------------
        # Step 3: 情绪识别
        # -----------------------------
        with st.spinner("正在进行情绪识别..."):
            try:
                if model:
                    result = model.infer_emotion_describe(
                        video_path=st.session_state.video_path,
                        audio_path=audio_path,
                        subtitle=subtitle_text
                    )
                    st.success("✅ 情绪识别完成")
                    st.subheader("情绪识别结果：")
                    st.markdown(result)
                else:
                    st.error("模型加载失败，请稍后重新加载网页")

            except Exception as e:
                st.error(f"情绪识别出错：{e}")

    if st.button("检测心率"):
        # -----------------------------
        # 心率检测（rPPG）
        # -----------------------------
        with st.spinner("正在检测心率..."):
            
            hr, img = analyze_heart_rate(st.session_state.video_path, gpu_id)
            st.success("✅ 心率检测完成")
            st.subheader("心率检测结果：")
            st.metric("估计心率", f"{hr:.2f} bpm")
            st.image(img, caption="rPPG 波形与功率谱", use_container_width=True)

except AttributeError:
    st.info("请先上传或拍摄一个视频。")
