import torch
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import MER2025OV_Dataset
from my_affectgpt.processors import BaseProcessor


# =======================================
# 模型封装类：AffectGPTInference
# =======================================
class AffectGPTInference:
    def __init__(self, cfg_path, ckpt_path, zeroshot=False, gpu_id=0):
        """初始化模型"""
        args = type('Args', (), {})()
        args.cfg_path = cfg_path
        args.options = None
        args.zeroshot = zeroshot
        args.outside_user_message = None
        args.outside_face_or_frame = None
        cfg = Config(args)

        self.model_cfg = cfg.model_cfg
        self.datasets_cfg = cfg.datasets_cfg
        self.inference_cfg = cfg.inference_cfg
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

        # 模型加载
        self.model_cfg.ckpt_3 = ckpt_path
        model_cls = registry.get_model_class(self.model_cfg.arch)
        self.model = model_cls.from_config(self.model_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.model = self.model.to(self.device).eval()

        self.chat = Chat(self.model, self.model_cfg, device=self.device)

        # 处理器与数据集
        self.dataset_cls = MER2025OV_Dataset()
        self.face_or_frame = self._get_face_or_frame()
        self.dataset_cls.vis_processor = BaseProcessor()
        self.dataset_cls.img_processor = BaseProcessor()
        self.dataset_cls.needed_data = self.dataset_cls.get_needed_data(self.face_or_frame)

        vis_processor_cfg = self.inference_cfg.get("vis_processor")
        img_processor_cfg = self.inference_cfg.get("img_processor")
        if vis_processor_cfg is not None:
            self.dataset_cls.vis_processor = registry.get_processor_class(
                vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
        if img_processor_cfg is not None:
            self.dataset_cls.img_processor = registry.get_processor_class(
                img_processor_cfg.train.name).from_config(img_processor_cfg.train)

        self.dataset_cls.n_frms = self.model_cfg.vis_processor.train.n_frms

        print(f"[INFO] AffectGPT 模型加载完成 ✅ 设备: {self.device}")

    def _get_face_or_frame(self):
        candidates = []
        if 'mercaptionplus' in self.datasets_cfg:
            candidates.append(self.datasets_cfg['mercaptionplus'].face_or_frame)
        if 'ovmerd' in self.datasets_cfg:
            candidates.append(self.datasets_cfg['ovmerd'].face_or_frame)
        assert len(set(candidates)) == 1, "face_or_frame 类型不一致"
        return candidates[0]

    def infer_emotion_ov(self, video_path, subtitle="", audio_path=None):
        """输入视频、音频、字幕，输出情感识别结果"""
        sample_data = self.dataset_cls.read_frame_face_audio_text(
            video_path, face_npy=None, audio_path=audio_path, image_path=None
        )

        # multimodal embedding
        audio_hiddens, audio_llms = self.chat.postprocess_audio(sample_data)
        frame_hiddens, frame_llms = self.chat.postprocess_frame(sample_data)
        face_hiddens, face_llms = self.chat.postprocess_face(sample_data)
        _, image_llms = self.chat.postprocess_image(sample_data)

        multi_llms = None
        if self.face_or_frame.startswith('multiface'):
            _, multi_llms = self.chat.postprocess_multi(face_hiddens, audio_hiddens)
        elif self.face_or_frame.startswith('multiframe'):
            _, multi_llms = self.chat.postprocess_multi(frame_hiddens, audio_hiddens)

        img_list = {
            'audio': audio_llms,
            'frame': frame_llms,
            'face': face_llms,
            'image': image_llms,
            'multi': multi_llms
        }

        user_message_ov = self.dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
        prompt_ov = self.dataset_cls.get_prompt_for_multimodal(self.face_or_frame, subtitle, user_message_ov)
        response_ov = self.chat.answer_sample(
            prompt=prompt_ov,
            img_list=img_list,
            num_beams=1,
            temperature=1,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=1200,
            max_length=2000
        )
        return response_ov

    def infer_emotion_describe(self, video_path, subtitle="", audio_path=None):
        """输入视频、音频、字幕，输出情感识别结果"""
        sample_data = self.dataset_cls.read_frame_face_audio_text(
            video_path, face_npy=None, audio_path=audio_path, image_path=None
        )

        # multimodal embedding
        audio_hiddens, audio_llms = self.chat.postprocess_audio(sample_data)
        frame_hiddens, frame_llms = self.chat.postprocess_frame(sample_data)
        face_hiddens, face_llms = self.chat.postprocess_face(sample_data)
        _, image_llms = self.chat.postprocess_image(sample_data)

        multi_llms = None
        if self.face_or_frame.startswith('multiface'):
            _, multi_llms = self.chat.postprocess_multi(face_hiddens, audio_hiddens)
        elif self.face_or_frame.startswith('multiframe'):
            _, multi_llms = self.chat.postprocess_multi(frame_hiddens, audio_hiddens)

        img_list = {
            'audio': audio_llms,
            'frame': frame_llms,
            'face': face_llms,
            'image': image_llms,
            'multi': multi_llms
        }
        user_message_describe = "Please infer the person's emotional state and provide your reasoning process."
        prompt_describe = self.dataset_cls.get_prompt_for_multimodal(self.face_or_frame, subtitle, user_message_describe)
        response_describe = self.chat.answer_sample(
            prompt=prompt_describe,
            img_list=img_list,
            num_beams=1,
            temperature=1,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=1200,
            max_length=2000
        )
        return response_describe