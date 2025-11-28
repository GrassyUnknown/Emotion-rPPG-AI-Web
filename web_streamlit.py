import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from aiortc.contrib.media import MediaRecorder
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import ffmpeg
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
st.set_page_config(page_title="结合生理信号的多模态情感识别系统", layout="wide")
st.title("🎥 结合生理信号的多模态情感识别系统")

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
    # Qwen 模型加载，gpu_memory_utilization与占用显存相关，目前总占用约66G
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname="Qwen25", gpu_memory_utilization=0.6)
    # Whisper 模型加载
    whisper_model = whisper.load_model("medium", f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    return model, llm, tokenizer, sampling_params, whisper_model

model, llm, tokenizer, sampling_params, whisper_model = load_model()
st.success("✅ 大模型已加载")

# 初始化会话状态
def init_session_state():
    defaults = {
        "uploaded_file": None,
        "result": None,
        "history": [],
        # 控制查看历史记录变量
        "view_history": False,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


init_session_state()
# 欢迎语
if not st.session_state.view_history:
    st.markdown("""
    该应用支持：
    - 上传或录制视频，视频中应有人脸；
    - 自动提取视频多模态信息（音频、字幕等）；
    - 使用rPPG识别技术分析心率和呼吸率；
    - 结合以上信息，使用大模型识别情感关键词和极性；
    - 结合以上信息，分析情感识别的原因；
    """)
else:
    st.markdown(f"正在查看历史记录{st.session_state.view_history_index + 1}")

class Result:
    def __init__(self, video_path=None, subtitle_text="", audio_path=None,
                 ov=None, ov_chi=None, sentiment=None, describe=None,
                 rppg_hr=None, rppg_img=None):
        self.video_path = video_path
        self.subtitle_text = subtitle_text
        self.audio_path = audio_path
        self.ov = ov
        self.ov_chi = ov_chi
        self.sentiment = sentiment
        self.describe = describe
        self.rppg_hr = rppg_hr
        self.rppg_img = rppg_img
        
def add_history():
    if st.session_state.result != None:
        print("Adding history for video path: " + st.session_state.result.video_path + ", subtitle: " + st.session_state.result.subtitle_text)
        st.session_state.history.append(st.session_state.result)
    else:
        print("Result is None, not adding history.")

def get_audio_path():
    if st.session_state.result.audio_path == None:
        try:
            with st.spinner("正在提取音频..."):
                video_clip = VideoFileClip(st.session_state.result.video_path)
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                video_clip.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
                video_clip.close()
                st.session_state.result.audio_path = temp_audio.name
            st.success("✅ 音频提取成功")
        except Exception as e:
            st.error("音频提取失败，可能是源视频中无音频信息。")
            st.session_state.result.audio_path = None
    if st.session_state.result.audio_path:
        st.audio(st.session_state.result.audio_path)

def get_subtitle_text():
    if st.session_state.result.subtitle_text == "":
        if st.session_state.result.audio_path == None:
            st.error("无法进行语音识别，缺少音频信息。")
            st.session_state.result.subtitle_text = ""
            return
        with st.spinner("正在识别音频..."):
            result = whisper_model.transcribe(st.session_state.result.audio_path, initial_prompt="接下来是一段视频的字幕。Here are subtitles of a video.")
            print("Result of whisper:" + result['text'])
            st.session_state.result.subtitle_text = result['text']
            # converter = opencc.OpenCC("t2s.json")
            # subtitle_text = converter.convert(subtitle_text)
            st.success("✅ 字幕识别成功")
    st.success("字幕：" + st.session_state.result.subtitle_text)
    if len(st.session_state.result.subtitle_text) > 125:
        with st.spinner("字幕信息过长，调用Qwen简化字幕内容..."):
            st.session_state.result.subtitle_text = subtitle_summarize_qwen(tokenizer, llm, sampling_params, 
                                                        subtitle=st.session_state.result.subtitle_text)
        st.success("✅ 字幕简化结果：" + st.session_state.result.subtitle_text)

# 展示Open-Vocabulary结果
def display_ov_result():
    raw = st.session_state.result.ov
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
    raw = st.session_state.result.ov_chi
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
    if st.session_state.result.ov == None:
        with st.spinner("正在进行情绪识别..."):
            try:
                result_ov = model.infer_emotion_ov(
                    video_path=st.session_state.result.video_path,
                    audio_path=st.session_state.result.audio_path,
                    subtitle=st.session_state.result.subtitle_text
                )
                print(result_ov)
                st.session_state.result.ov = reason_to_openset_qwen(tokenizer, llm, sampling_params, result_ov)
                st.session_state.result.ov_chi = reason_to_openset_qwen_chi(tokenizer, llm, sampling_params, result_ov)
            except Exception as e:
                st.error(f"情绪识别出错：{e}")
        st.success("✅ 情绪识别完成")
    st.subheader("情绪关键词：")
    display_ov_result()
    display_ov_result_chi()

def get_emotion_result_sentiment():
    if st.session_state.result.sentiment is None:
        with st.spinner("正在进行情绪极性识别..."):
            polarity = float(reason_to_valence_qwen(tokenizer, llm, sampling_params, reason=st.session_state.result.ov))
            fig, ax = plt.subplots(figsize=(6, 1))
            gradient = np.linspace(0, 1, 500).reshape(1, -1)
            ax.imshow(gradient, extent=(-1, 1, 0.4, 0.6),
                    cmap="rainbow", aspect="auto")  # 蓝-白-红
            ax.hlines(0.5, -1, 1, color="black", linewidth=1)
            ax.plot(polarity, 0.5, "o", markersize=16, color="black")
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticks([])
            ax.text(polarity, 0.7, f"{polarity:.2f}", ha="center", fontsize=12)
            ax.text(-1, 0.25, "😡 Negative", ha="center", fontsize=12, color="blue")
            ax.text(0, 0.25, "😐 Neutral", ha="center", fontsize=12, color="gray")
            ax.text(1, 0.25, "😄 Positive", ha="center", fontsize=12, color="red")
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.session_state.result.sentiment = fig
        st.success("✅ 情绪极性识别完成")
    st.subheader("情绪极性：")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(st.session_state.result.sentiment, use_container_width=False)

def get_emotion_result_describe():
    if st.session_state.result.describe == None:
        with st.spinner("正在分析..."):
            try:
                result_describe = model.infer_emotion_describe(
                    video_path=st.session_state.result.video_path,
                    audio_path=st.session_state.result.audio_path,
                    subtitle=st.session_state.result.subtitle_text
                )
                print(result_describe)
                st.markdown(f"初步分析：{result_describe}")
                st.session_state.result.describe = reason_merge_qwen(tokenizer, llm, sampling_params, 
                                                                reason=result_describe,
                                                                subtitle=st.session_state.result.subtitle_text,
                                                                hr=st.session_state.result.rppg_hr,
                                                                ov=st.session_state.result.ov)
                st.session_state.result.describe += "\n\n"
                st.session_state.result.describe += translate_eng2chi_qwen(tokenizer, llm, sampling_params, 
                                                                reason=st.session_state.result.describe)
            except Exception as e:
                st.error(f"情绪分析出错：{e}")
        st.success("✅ 情绪分析完成")
    st.subheader("情绪分析结果：")
    st.markdown(st.session_state.result.describe)

def get_rppg():
    if st.session_state.result.rppg_hr == None:
        with st.spinner("正在检测..."):            
            st.session_state.result.rppg_hr, st.session_state.result.rppg_img = \
            analyze_heart_rate(st.session_state.result.video_path, gpu_id)
        st.success("✅ 生理信号检测完成")
    st.subheader("生理信号检测结果：")
    st.metric("估计心率", f"{st.session_state.result.rppg_hr:.2f} bpm")
    st.image(st.session_state.result.rppg_img, caption="波形图", use_container_width=True)


if not st.session_state.view_history:
    # 上传或拍摄视频
    option = st.radio("选择输入方式：", ["上传视频文件", "使用摄像头拍摄"])
    if option == "上传视频文件":
        uploaded_file = st.file_uploader("请上传视频文件", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file != None and st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            if uploaded_file.type != "video/mp4":
                video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                ffmpeg.input(temp_file.name).output(video_path, vcodec='libx264', acodec='aac').run(overwrite_output=True)
            else:
                video_path = temp_file.name
            add_history()
            st.session_state.result = Result(video_path=video_path)
    elif option == "使用摄像头拍摄":
        st.markdown("录制说明：点击下方的“START”按钮，允许访问摄像头后即开始录制视频。录制完成后点击”STOP“，待加载完成后再点击“完成录制”按钮，以获取录制的视频。")
        def recorder_factory() -> MediaRecorder:
            return MediaRecorder('/tmp/record.mp4' , format="mp4")
        # 启动 WebRTC 以录制
        webrtc_streamer(
            key="record_only",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": True},  # 启用音视频
            in_recorder_factory=recorder_factory,
        )
        if st.button("完成录制"):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            with open('/tmp/record.mp4', "rb") as record:
                temp_file.seek(0)
                temp_file.truncate()
                temp_file.write(record.read())
            video_path=temp_file.name
            add_history()
            st.session_state.result = Result(video_path=video_path)
    # 展示视频
if st.session_state.result != None and st.session_state.result.video_path != None:
    st.video(st.session_state.result.video_path)


if not st.session_state.view_history:
    # 字幕输入
    st.subheader("💬 视频里的人说了什么？")
    subtitle_text = st.text_area("请输入字幕（可选）", placeholder="若不输入，将自动进行语音识别。注意：错误的输入将显著影响识别结果。", height=100)
    # 用户输入了字幕
    if st.session_state.result != None:
        if subtitle_text != "":
            print("User input subtitle: " + subtitle_text)
            st.session_state.user_subtitle_text = subtitle_text
            # 存储的字幕信息与输入不一致，则更新
            if st.session_state.result.subtitle_text != st.session_state.user_subtitle_text:
                add_history()
                st.session_state.result = Result(
                    video_path=st.session_state.result.video_path,
                    audio_path=st.session_state.result.audio_path,
                    subtitle_text=st.session_state.user_subtitle_text
                )
        else:
            # 本次未输入字幕，但用户上一次输入了字幕
            if hasattr(st.session_state, "user_subtitle_text"):
                del st.session_state.user_subtitle_text
                add_history()
                st.session_state.result = Result(
                    video_path=st.session_state.result.video_path,
                    audio_path=st.session_state.result.audio_path,
                )
            # 未输入字幕，且上一次也未输入字幕，则继续使用语音识别结果        


if st.session_state.result != None and st.session_state.result.video_path != None:
    if st.button("检测生理信号"):
        get_rppg()
    if st.button("分析情绪关键词"):
        get_audio_path()
        get_subtitle_text()
        get_emotion_result_ov()
        get_emotion_result_sentiment()
    if st.button("描述情绪"):
        get_audio_path()
        get_subtitle_text()
        get_emotion_result_describe()
else:
    st.info("请先上传或拍摄一条视频。")

# 点击侧边栏按钮的行为：若正在查看历史记录，则保存当前记录；否则新增历史记录
def click_sidebar_button():
    if (not st.session_state.view_history):
        add_history()
    elif st.session_state.view_history:
        st.session_state.history[st.session_state.view_history_index] = st.session_state.result

# 历史记录栏
with st.sidebar:
    if st.button("新建分析"):
        click_sidebar_button()
        st.session_state.view_history = False
        st.session_state.result = None
        st.rerun()

    st.title("💬 历史记录")
    if(len(st.session_state.history) == 0):
        st.write("暂无历史记录。")
    for i in range(len(st.session_state.history)-1, -1, -1):
        if st.button(f"记录 {i+1} " + st.session_state.history[i].subtitle_text):
            click_sidebar_button()
            st.session_state.view_history = True
            st.session_state.view_history_index = i
            st.session_state.result = st.session_state.history[i]
            st.rerun()
    