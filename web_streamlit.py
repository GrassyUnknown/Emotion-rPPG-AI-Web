import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from aiortc.contrib.media import MediaRecorder
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

# AffectGPT 推理依赖
sys.path.append(os.path.join(os.path.dirname(__file__), "AffectGPT"))
from affectgpt_inference import AffectGPTInference

# RPPG 推理依赖
from rppg.demo import analyze_heart_rate

# Qwen 推理依赖
from qwen import *

gpu_id = 0

# Streamlit
st.set_page_config(page_title="情感识别与心率检测", layout="wide")
st.title("🎥 多模态情感识别与心率检测系统")

# 持久缓存模型
@st.cache_resource(show_spinner=True)
def load_model():
    # AffectGPT 模型加载
    model = AffectGPTInference(
        cfg_path="/home/zhangzijie/Emotion-rPPG-AI-Web/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml",
        ckpt_path="/home/zhangzijie/Emotion-rPPG-AI-Web/AffectGPT/models/AffectGPT/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth",
        zeroshot=True,
        gpu_id=gpu_id
    )
    # Qwen 模型加载
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname="Qwen25")
    # Whisper 模型加载
    whisper_model = whisper.load_model("medium", f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    return model, llm, tokenizer, sampling_params, whisper_model

try:
    model, llm, tokenizer, sampling_params, whisper_model = load_model()
    st.success("✅ 大模型已加载")
except Exception as e:
    st.error(f"⚠️ 模型加载失败：{e}")
    model = None

st.markdown("""
该应用支持：
- 上传或录制视频；
- 自动提取视频音频；
- 使用 AffectGPT 模型分析情感状态；
- 使用 Contrast-Phys 分析心率状态；
""")

# 初始化会话状态
def init_session_state():
    defaults = {
        "uploaded_file": None,
        "video_path": "",
        "subtitle_text": "",
        "audio_path": "",
        "result_ov": "",
        "result_ov_chi": "",
        "result_describe": "",
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)
init_session_state()

def get_audio_path():
    if st.session_state.audio_path == "":
        with st.spinner("正在提取音频..."):
            video_clip = VideoFileClip(st.session_state.video_path)
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            video_clip.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
            video_clip.close()
            st.session_state.audio_path = temp_audio.name
    if st.session_state.audio_path:
        st.success("✅ 音频提取成功")
        st.audio(st.session_state.audio_path)

def get_subtitle_text():
    if st.session_state.subtitle_text == "":
        with st.spinner("正在识别音频..."):
            result = whisper_model.transcribe(st.session_state.audio_path, initial_prompt="接下来是一段视频的字幕。Here are subtitles of a video.")
            print("Result of whisper:" + result['text'])
            st.session_state.subtitle_text = result['text']
            # converter = opencc.OpenCC("t2s.json")
            # subtitle_text = converter.convert(subtitle_text)
    st.success("✅ 字幕：" + st.session_state.subtitle_text)
    if len(st.session_state.subtitle_text) > 125:
        with st.spinner("字幕信息过长，调用Qwen简化字幕内容..."):
            st.session_state.subtitle_text = subtitle_summarize_qwen(tokenizer, llm, sampling_params, 
                                                        subtitle=st.session_state.subtitle_text)
        st.success("✅ 字幕简化结果：" + st.session_state.subtitle_text)

# 展示Open-Vocabulary结果
def display_ov_result():
    raw = st.session_state.result_ov
    emotions = [e.strip() for e in raw.strip("[]").split(",") if e.strip()]

    if emotions:
        colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#6a4c93"]
        tags = " ".join(
            f"<span style='font-size:26px;font-weight:700;color:{colors[i%5]};margin-right:12px;'>{e}</span>"
            for i, e in enumerate(emotions)
        )
        st.markdown(f"<div style='margin:12px 0;'>{tags}</div>", unsafe_allow_html=True)
    else:
        st.write("未识别出英文情绪关键词。")

def display_ov_result_chi():
    raw = st.session_state.result_ov_chi
    emotions = [e.strip() for e in re.split(r"[、,，]", raw.strip("[]")) if e.strip()]

    if emotions:
        colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#6a4c93"]
        tags = " ".join(
            f"<span style='font-size:26px;font-weight:700;color:{colors[i%5]};margin-right:12px;'>{e}</span>"
            for i, e in enumerate(emotions)
        )
        st.markdown(f"<div style='margin:12px 0;'>{tags}</div>", unsafe_allow_html=True)
    else:
        st.write("未识别出中文情绪关键词。")

def get_emotion_result_ov():
    if st.session_state.result_ov == "":
        with st.spinner("正在进行情绪识别..."):
            try:
                result_ov = model.infer_emotion_ov(
                    video_path=st.session_state.video_path,
                    audio_path=st.session_state.audio_path,
                    subtitle=st.session_state.subtitle_text
                )
                print(result_ov)
                st.session_state.result_ov = reason_to_openset_qwen(tokenizer, llm, sampling_params, result_ov)
                st.session_state.result_ov_chi = reason_to_openset_qwen_chi(tokenizer, llm, sampling_params, result_ov)
            except Exception as e:
                st.error(f"情绪识别出错：{e}")
    st.success("✅ 情绪识别完成")
    st.subheader("情绪识别结果：")
    display_ov_result()
    display_ov_result_chi()

def get_emotion_result_describe():
    if st.session_state.result_describe == "":
        with st.spinner("正在进行情绪识别..."):
            try:
                result_describe = model.infer_emotion_describe(
                    video_path=st.session_state.video_path,
                    audio_path=st.session_state.audio_path,
                    subtitle=st.session_state.subtitle_text
                )
                print(result_describe)
                st.session_state.result_describe = reason_merge_qwen(tokenizer, llm, sampling_params, 
                                                                reason=result_describe,
                                                                subtitle=st.session_state.subtitle_text)
                st.session_state.result_describe += "\n\n"
                st.session_state.result_describe += translate_eng2chi_qwen(tokenizer, llm, sampling_params, 
                                                                reason=st.session_state.result_describe)
            except Exception as e:
                st.error(f"情绪识别出错：{e}")
    st.success("✅ 情绪识别完成")
    st.subheader("情绪识别结果：")
    st.markdown(st.session_state.result_describe)

# 当上传新视频时，清空结果
def clear_session_state_with_new_video():
    st.session_state.subtitle_text = ""
    st.session_state.audio_path = ""
    st.session_state.result_ov = ""
    st.session_state.result_describe = ""

# 上传或拍摄视频
option = st.radio("选择输入方式：", ["上传视频文件", "使用摄像头拍摄"])
if option == "上传视频文件":
    uploaded_file = st.file_uploader("请上传视频文件（mp4 / mov / avi）", type=["mp4", "mov", "avi"])
    if uploaded_file != None and st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        st.session_state.video_path = temp_file.name
        clear_session_state_with_new_video()
elif option == "使用摄像头拍摄":
    def recorder_factory() -> MediaRecorder:
        return MediaRecorder('/tmp/record.mp4' , format="mp4")
    # 启动 WebRTC 以录制
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
        clear_session_state_with_new_video()
# 展示视频
if st.session_state.video_path != "":
    st.video(st.session_state.video_path)


# 字幕输入
st.subheader("💬 视频里的人说了什么？")
subtitle_text = st.text_area("请输入字幕（可选）", placeholder="若不输入，将自动进行语音识别。注意：错误的输入将显著影响识别结果。", height=100)
# 用户输入了字幕
if subtitle_text != "":
    print("User input subtitle: " + subtitle_text)
    st.session_state.user_subtitle_text = subtitle_text
    # 存储的字幕信息与输入不一致，则更新
    if st.session_state.subtitle_text != st.session_state.user_subtitle_text:
        st.session_state.subtitle_text = st.session_state.user_subtitle_text
        st.session_state.result_ov = ""
        st.session_state.result_describe = ""
else:
    # 本次未输入字幕，但用户上一次输入了字幕
    if hasattr(st.session_state, "user_subtitle_text"):
        del st.session_state.user_subtitle_text
        st.session_state.subtitle_text = ""
        st.session_state.result_ov = ""
        st.session_state.result_describe = ""
    # 未输入字幕，且上一次也未输入字幕，则继续使用语音识别结果        


if st.session_state.video_path != "":
    if st.button("分析情绪关键词"):
        get_audio_path()
        get_subtitle_text()
        get_emotion_result_ov()
    if st.button("描述情绪"):
        get_audio_path()
        get_subtitle_text()
        get_emotion_result_describe()
    if st.button("检测心率"):
        with st.spinner("正在检测心率..."):            
            hr, img = analyze_heart_rate(st.session_state.video_path, gpu_id)
            st.success("✅ 心率检测完成")
            st.subheader("心率检测结果：")
            st.metric("估计心率", f"{hr:.2f} bpm")
            st.image(img, caption="rPPG 波形与功率谱", use_container_width=True)
else:
    st.info("请先上传或拍摄一条视频。")

#TODO: 增加历史记录
with st.sidebar:
    st.title("💬 历史记录")
    st.markdown("_功能开发中，敬请期待！_")