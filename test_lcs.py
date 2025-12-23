import os
import time
import torch
import numpy as np
import difflib
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

# Use non-interactive backend for cluster execution
matplotlib.use('Agg')

# -----------------------------
# 1. Configuration
# -----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Note: You must have approved access to Llama-3.1-70B on Hugging Face
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"

MAX_ITERATIONS = 3
LCS_THRESHOLD = 0.85

# -----------------------------
# 2. Initialization
# -----------------------------
print("Loading metrics...")
chrf_metric = evaluate.load("chrf")
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")
comet_metric = evaluate.load("comet")

class TranslationEvaluator:
    def __init__(self):
        # Ensure HUGGING_FACE_HUB_TOKEN is defined at the top of your script
        # from the os.getenv call you already have.
        
        # 1. Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # 2. Loading Llama-3.1-70B (PASS THE TOKEN HERE)
        print(f"Loading {LLAMA_MODEL_ID} (4-bit)...")
        self.llama_tok = AutoTokenizer.from_pretrained(
            LLAMA_MODEL_ID, 
            token=HUGGING_FACE_HUB_TOKEN  # Add this line
        )
        self.llama_mod = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            token=HUGGING_FACE_HUB_TOKEN, # Add this line
            trust_remote_code=True
        )
        
        # 3. Loading mBART-50
        print(f"Loading mBART-50...")
        self.nmt_tok = AutoTokenizer.from_pretrained(
            MBART_MODEL_ID,
            token=HUGGING_FACE_HUB_TOKEN # Recommended for consistency
        )
        self.nmt_mod = AutoModelForSeq2SeqLM.from_pretrained(
            MBART_MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="auto",
            token=HUGGING_FACE_HUB_TOKEN # Recommended for consistency
        )

    def mbart_translate(self, text, src_iso, tgt_iso):
        mapping = {"eng": "en_XX", "npi": "ne_NP"}
        self.nmt_tok.src_lang = mapping[src_iso]
        inputs = self.nmt_tok(text, return_tensors="pt", padding=True).to(self.nmt_mod.device)
        outputs = self.nmt_mod.generate(**inputs, forced_bos_token_id=self.nmt_tok.lang_code_to_id[mapping[tgt_iso]])
        return self.nmt_tok.decode(outputs[0], skip_special_tokens=True)

    def llama_generate(self, messages):
        prompt = self.llama_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        
        with torch.no_grad():
            outputs = self.llama_mod.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=False,
                pad_token_id=self.llama_tok.eos_token_id
            )
        
        response_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return self.llama_tok.decode(response_ids, skip_special_tokens=True).strip()

    def llama_direct(self, source):
        messages = [
            {"role": "system", "content": "You are a professional English to Nepali translator."},
            {"role": "user", "content": f"Translate this to Nepali. Output only the translation: {source}"}
        ]
        return self.llama_generate(messages)

    def llama_refine(self, source, draft, back_trans):
        messages = [
            {"role": "system", "content": "You are an expert editor. Fix Nepali translations based on back-translation errors."},
            {"role": "user", "content": (
                f"Original: {source}\nDraft: {draft}\nBack-translation errors: {back_trans}\n\n"
                "Provide the corrected Nepali translation. Output only the text."
            )}
        ]
        return self.llama_generate(messages)

# -----------------------------
# 3. Execution
# -----------------------------
def execute(num_samples=50):
    evaluator = TranslationEvaluator()
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    
    src_df = df[df['iso_639_3'] == 'eng'].reset_index()
    tgt_df = df[df['iso_639_3'] == 'npi'].reset_index()

    data_store = {"mBART": [], "Llama_Direct": [], "LLM_BTI": [], "Ref": [], "Src": []}

    print(f"Starting eval on {num_samples} samples...")
    for i in range(num_samples):
        src_text = src_df.iloc[i]['text']
        ref_text = tgt_df.iloc[i]['text']

        # 1. mBART
        mbart_res = evaluator.mbart_translate(src_text, "eng", "npi")
        
        # 2. Llama Direct
        llama_dir_res = evaluator.llama_direct(src_text)
        
        # 3. Iterative BTI
        current = mbart_res
        for _ in range(MAX_ITERATIONS):
            back = evaluator.mbart_translate(current, "npi", "eng")
            sim = difflib.SequenceMatcher(None, src_text.split(), back.split()).ratio()
            if sim >= 0.90: break
            current = evaluator.llama_refine(src_text, current, back)
        
        data_store["mBART"].append(mbart_res)
        data_store["Llama_Direct"].append(llama_dir_res)
        data_store["LLM_BTI"].append(current)
        data_store["Ref"].append(ref_text)
        data_store["Src"].append(src_text)
        
        if (i+1) % 5 == 0:
            print(f"Progress: {i+1}/{num_samples}")

    # -----------------------------
    # 4. Scoring & Graphing
    # -----------------------------
    final_scores = {}
    metrics_list = ["chrF", "BLEU", "BERTScore", "COMET"]
    
    for model in ["mBART", "Llama_Direct", "LLM_BTI"]:
        preds = data_store[model]
        refs = [[r] for r in data_store["Ref"]]
        refs_flat = data_store["Ref"]
        srcs = data_store["Src"]

        ch = chrf_metric.compute(predictions=preds, references=refs)['score']
        bl = bleu_metric.compute(predictions=preds, references=refs)['bleu'] * 100
        bs = np.mean(bertscore_metric.compute(predictions=preds, references=refs_flat, lang="ne")['f1']) * 100
        cm = comet_metric.compute(predictions=preds, references=refs_flat, sources=srcs)['mean_score'] * 100
        
        final_scores[model] = [ch, bl, bs, cm]

    # Print Result Table
    print("\n" + "="*60)
    print(f"{'Method':<15} | {'chrF':<6} | {'BLEU':<6} | {'BERT':<6} | {'COMET':<6}")
    print("-" * 60)
    for model, s in final_scores.items():
        print(f"{model:<15} | {s[0]:<6.2f} | {s[1]:<6.2f} | {s[2]:<6.2f} | {s[3]:<6.2f}")
    print("="*60)

    # Generate Plot
    x = np.arange(len(metrics_list))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - width, final_scores["mBART"], width, label='mBART (Baseline)', color='#1f77b4')
    ax.bar(x, final_scores["Llama_Direct"], width, label='Llama-3.1-70B (Direct)', color='#ff7f0e')
    ax.bar(x + width, final_scores["LLM_BTI"], width, label='LLM-BTI (Iterative)', color='#2ca02c')

    ax.set_ylabel('Scores (0-100 scale)')
    ax.set_title(f'Comparison: English to Nepali Translation (N={num_samples})')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.savefig('llama70b_bti_results.png', dpi=300)
    print("Graph saved as 'llama70b_bti_results.png'")

if __name__ == "__main__":
    execute(num_samples=5)
