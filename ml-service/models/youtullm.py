import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- 1. Load model và tokenizer Youtu-LLM (chỉ 1 lần) ---
MODEL_ID = "tencent/Youtu-LLM-2B"

print("Đang tải Youtu-LLM, có thể mất vài phút...")
yt_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
yt_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True
)
print("Youtu-LLM đã sẵn sàng!")

# --- 2. Hàm parse tư duy + trả lời ---
def parse_reasoning(text: str):
    """Trích <think>…</think> nếu có, và phần còn lại là câu trả lời"""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        answer = text.split("</think>")[-1].strip()
    else:
        thought = "(Không có quá trình tư duy rõ ràng)"
        answer = text
    return thought, answer

# --- 3. Hàm mới cho Youtu-LLM ---
def generate_youtu(prompt: str, max_new_tokens=512):
    """
    Sinh câu trả lời từ Youtu-LLM
    - prompt: câu hỏi hoặc lệnh
    - max_new_tokens: token tối đa
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = yt_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # bật Reasoning Mode
    )

    model_inputs = yt_tokenizer([input_text], return_tensors="pt").to(yt_model.device)

    outputs = yt_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_k=20,
        top_p=0.95,
        repetition_penalty=1.05
    )

    full_response = yt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    thought, answer = parse_reasoning(full_response)

    return answer
