from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "AffectGPT"))
import config


def func_postprocess_qwen(response):
    response = response.strip()
    if response.startswith("输入"):   response = response[len("输入"):]
    if response.startswith("输出"):   response = response[len("输出"):]
    if response.startswith("翻译"):   response = response[len("翻译"):]
    if response.startswith("让我们来翻译一下："): response = response[len("让我们来翻译一下："):]
    if response.startswith("output"): response = response[len("output"):]
    if response.startswith("Output"): response = response[len("Output"):]
    if response.startswith("input"): response = response[len("input"):]
    if response.startswith("Input"): response = response[len("Input"):]
    response = response.strip()
    if response.startswith(":"):  response = response[len(":"):]
    if response.startswith("："): response = response[len("："):]
    response = response.strip()
    # response = response.replace('\n', '') # remove \n
    response = response.strip()
    return response

def func_read_batch_calling_model(modelname, gpu_memory_utilization=0.4):
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params

# ========================= # 
##      基本操作：翻译      ##
# ========================= # 
def translate_chi2eng_qwen(tokenizer=None, llm=None, sampling_params=None, reason=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please translate the Chinese input into English. Please ensure the translated results does not contain any Chinese words.
Input: 高兴; Output: happy \
Input: 生气; Output: angry \
Input: {reason}; Output: """
        return prompt

    prompt = func_prompt_template(reason)
    prompt_list = [prompt]
    response_list = get_completion_qwen_batch(llm, sampling_params, tokenizer, prompt_list)
    return response_list[0]


def translate_eng2chi_qwen(tokenizer=None, llm=None, sampling_params=None, reason=None):
    
    def func_prompt_template(reason):
        prompt = f"""将以下英文翻译成中文。{reason}"""
        return prompt

    prompt = func_prompt_template(reason)
    prompt_list = [prompt]
    response_list = get_completion_qwen_batch(llm, sampling_params, tokenizer, prompt_list)
    return response_list[0]

# 依赖于 vllm 
def get_completion_qwen_batch(llm, sampling_params, tokenizer, prompt_list):
    
    assert isinstance(prompt_list, list)

    message_batch = []
    for prompt in prompt_list:
        message_batch.append([{"role": "user", "content": prompt}])

    text_batch = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(text_batch, sampling_params)
    
    # => batch_responses
    batch_responses = []
    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        response = func_postprocess_qwen(response)
        batch_responses.append(response)
        print(f"Prompt: {prompt} \n Response: {response}")
    return batch_responses

def subtitle_summarize_qwen(tokenizer=None, llm=None, sampling_params=None, subtitle=None):
    
    def func_prompt_template(subtitle):
        prompt = f"""Please act as an expert in text summarization. \
We provide you with subtitle content from a video. Your task is to condense this subtitle information into a more concise form while retaining the original meaning. \
Please ensure that the summarized content is clear and easy to understand. \
Your output should be less than 100 characters. \
Input: {subtitle}; Output: """
        return prompt
    
    prompt = func_prompt_template(subtitle)
    prompt_list = [prompt]
    response_list = get_completion_qwen_batch(llm, sampling_params, tokenizer, prompt_list)
    return response_list[0]

## reason -> ov labels
def reason_to_openset_qwen(tokenizer=None, llm=None, sampling_params=None, reason=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please assume the role of an expert in the field of emotions. \
We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the speaker. \
Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
If none are identified, please output an empty list. \
Input: We cannot recognize his emotional state; Output: [] \
Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] \
Input: {reason}; Output: """
        return prompt
    
    
    prompt = func_prompt_template(reason)
    prompt_list = [prompt]
    response_list = get_completion_qwen_batch(llm, sampling_params, tokenizer, prompt_list)
    return response_list[0]

## reason -> ov labels chi
def reason_to_openset_qwen_chi(tokenizer=None, llm=None, sampling_params=None, reason=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please assume the role of an expert in the field of emotions. \
We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the speaker. \
Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
If none are identified, please output an empty list. \
回复的结果请使用中文。
Input: We cannot recognize his emotional state; Output: [] \
Input: His emotional state is happy, sad, and angry; Output: [开心，伤心，生气] \
Input: {reason}; Output: """
        return prompt
    
    
    prompt = func_prompt_template(reason)
    prompt_list = [prompt]
    response_list = get_completion_qwen_batch(llm, sampling_params, tokenizer, prompt_list)
    return response_list[0]

def reason_to_valence_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please identify the overall positive or negative emotional polarity of the speaker.  \
The output should be a ﬂoating-point number ranging from -1 to 1.  \
Here, -1 indicates extremely negative emotions, 0 indicates neutral emotions, and 1 indicates extremely positive emotions.  \
Please provide your judgment as a ﬂoating-point number.  \
Input: I am very happy; Output: 1  \
Input: I am very angry; Output: -1 \
Input: I am neutral; Output: 0 \
Input: {reason}; Output: """
        return prompt
    
    prompt = func_prompt_template(reason)
    prompt_list = [prompt]
    response_list = get_completion_qwen_batch(llm, sampling_params, tokenizer, prompt_list)
    return response_list[0]

def reason_merge_qwen(tokenizer=None, llm=None, sampling_params=None, 
                      reason=None, subtitle=None):
    
    def func_prompt_template(reason, subtitle):
        
        reason_merge = ""
        reason_merge += f"Video and audio clue: {reason}；"
        reason_merge += f"Speaker says(Subtitle): {subtitle}"
        prompt = f"Please assume the role of an expert in the field of emotions. \
    We have provided clues from the video that may be related to the speaker's emotional states. \
    In addition, we have also provided the subtitle content of the video. \
    Please merge all these information to infer the emotional states of the speaker, and provide reasoning for your inferences. \
    Input: {reason_merge}\
    Output:"
        return prompt

    prompt = func_prompt_template(reason, subtitle)
    prompt_list = [prompt]
    response_list = get_completion_qwen_batch(llm, sampling_params, tokenizer, prompt_list)
    return response_list[0]


if __name__ == "__main__":
    # test
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname="Qwen25")
    reason = "In the text, the caption reads: \"Before we proceed, let me check something. I need to know my audience, how many of you guys here use TikTok.\" This sentence could be part of a speaker's introduction or survey conducted in a classroom or meeting setting. Given the video hints of the speaker's enthusiastic and passionate demeanor, and the audio cues of the speaker's clear, confident, and authoritative voice, we can infer that the speaker is interested in the audience's social networking behavior and hopes to better understand their audience's preferences through this survey. Therefore, this sentence expresses the speaker's excitement and anticipation, which aligns with the positive emotional state described in both the video and audio clues."
    subtitle = "Before we do that, can I just check, I need to know my audience. How many of you here use TikTok?"
    response1 = reason_to_openset_qwen(tokenizer=tokenizer, llm=llm, sampling_params=sampling_params, reason=reason)
    response2 = reason_to_valence_qwen(tokenizer=tokenizer, llm=llm, sampling_params=sampling_params, reason=reason)
    response3 = reason_merge_qwen(tokenizer=tokenizer, llm=llm, sampling_params=sampling_params, reason=reason, subtitle=subtitle)
    print("Openset Response:", response1)
    print("Valence Response:", response2)
    print("Merge Response:", response3)