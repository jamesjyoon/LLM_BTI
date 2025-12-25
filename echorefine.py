import os
import torch
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
import evaluate
import sacrebleu
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig
)
from comet import download_model, load_from_checkpoint

# --- Cluster Setup ---
import matplotlib
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# -----------------------------
# 1. Configuration
# -----------------------------
LLAMA_ID = "meta-llama/Llama-3.1-70B-Instruct" 
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
INTERNAL_QE_ID = "Unbabel/wmt22-cometkiwi-da"
FINAL_COMET_ID = "Unbabel/wmt22-comet-da"

EXPERIMENT_CONFIG = [
    {"iso": "npi", "name": "Nepali", "mbart": "ne_NP", "tier": "Low-Res"},
    {"iso": "ben", "name": "Bengali", "mbart": "bn_IN", "tier": "Mid-Res"},
    {"iso": "sin", "name": "Sinhala", "mbart": "si_LK", "tier": "Low-Res"},
    {"iso": "mya", "name": "Burmese", "mbart": "my_MM", "tier": "Low-Res"},
    {"iso": "kor", "name": "Korean", "mbart": "ko_KR", "tier": "Qualitative"},
]

# -----------------------------
# 2. System Architecture
# -----------------------------
class EchoRefineSystem:
    def __init__(self):
        # 4-bit config to reduce spikes
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        
        print(">>> Step 1: Loading Llama-3.1-70B (Refiner)...")
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID)
        self.l_mod = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto"
        )
        
        print(">>> Step 2: Loading mBART-50 (Drafter)...")
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(
            MBART_ID, torch_dtype=torch.float16, device_map="auto"
        )

        print(">>> Step 3: Loading CometKiwi (Judge)...")
        qe_path = download_model(INTERNAL_QE_ID)
        self.qe_mod = load_from_checkpoint(qe_path).to("cuda")

    def mbart_translate(self, text, src_tag, tgt_tag):
        self.n_tok.src_lang = src_tag
        inputs = self.n_tok(text, return_tensors="pt", truncation=True).to(self.n_mod.device)
        outputs = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt_tag])
        return self.n_tok.decode(outputs[0], skip_special_tokens=True)

    def refine_with_cot(self, source, draft, back_trans, lang_name):
        messages = [
            {"role": "system", "content": f"You are a professional {lang_name} translator. Your task is to fix the translation draft based on back-translation discrepancies. Provide ONLY the corrected text after 'RESULT:'."},
            {"role": "user", "content": f"Original English: {source}\n{lang_name} Draft: {draft}\nBack-translation into English: {back_trans}\n\nRESULT:"}
        ]
        prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.l_mod.device)
        outputs = self.l_mod.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=self.l_tok.eos_token_id)
        res = self.l_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        final = res.split("RESULT:")[-1].strip() if "RESULT:" in res else res
        # Language Guard
        if len(re.findall(r'[a-zA-Z]', final)) > (len(final) * 0.25): return draft
        return final

    def judge(self, source, draft, refined):
        data = [{"src": source, "mt": draft}, {"src": source, "mt": refined}]
        with torch.no_grad():
            scores = self.qe_mod.predict(data, batch_size=2, gpus=1, progress_bar=False).scores
        
        # Selection Logic: Margin of 0.02 to favor LLM fluency
        if scores[1] > (scores[0] - 0.02):
            return refined, "LLM"
        return draft, "mBART"

# -----------------------------
# 3. Main Execution
# -----------------------------
def execute_research():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default=None, help='ISO code (e.g. npi)')
    args, _ = parser.parse_known_args()

    configs = EXPERIMENT_CONFIG
    if args.lang:
        configs = [c for c in EXPERIMENT_CONFIG if c['iso'] == args.lang]

    # Initialize System (Loads models)
    sys = EchoRefineSystem()
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    
    # Load Evaluation Metrics
    print(">>> Loading Evaluation Metrics...")
    chrf = evaluate.load("chrf")
    comet_ref = evaluate.load("comet", FINAL_COMET_ID)

    final_report = []

    for cfg in configs:
        print(f"\n--- Processing: {cfg['name']} ---")
        src_texts = df[df['iso_639_3'] == 'eng']['text'].tolist()[:50] # Sample N=50
        ref_texts = df[df['iso_639_3'] == cfg['iso']]['text'].tolist()[:50]
        
        storage = {"mBART": [], "EchoRefine": []}
        selection_counts = {"LLM": 0, "mBART": 0}

        for i in tqdm(range(len(src_texts))):
            source = src_texts[i]
            # 1. Pipeline
            draft = sys.mbart_translate(source, "en_XX", cfg['mbart'])
            back = sys.mbart_translate(draft, cfg['mbart'], "en_XX")
            refined = sys.refine_with_cot(source, draft, back, cfg['name'])
            # 2. Judge
            final, winner = sys.judge(source, draft, refined)

            storage["mBART"].append(draft)
            storage["EchoRefine"].append(final)
            selection_counts[winner] += 1

        print(f"Selection Rate: {selection_counts}")

        # --- Compute Scores ---
        row = {"Language": cfg['name']}
        for key in ["mBART", "EchoRefine"]:
            preds = storage[key]
            refs = [[r] for r in ref_texts]
            
            row[f"{key}_BLEU"] = sacrebleu.corpus_bleu(preds, refs).score
            row[f"{key}_chrF"] = chrf.compute(predictions=preds, references=refs)['score']
            # Note: BERTScore removed to save RAM during large model runs
            row[f"{key}_COMET"] = comet_ref.compute(predictions=preds, references=ref_texts, sources=src_texts)['mean_score'] * 100
        
        final_report.append(row)

    # -----------------------------
    # 4. Result Graphing
    # -----------------------------
    results_df = pd.DataFrame(final_report)
    results_df.to_csv(f"results_{args.lang or 'all'}.csv", index=False)
    
    metrics = ['BLEU', 'chrF', 'COMET']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, m in enumerate(metrics):
        results_df.plot(x="Language", y=[f"mBART_{m}", f"EchoRefine_{m}"], 
                        kind="bar", ax=axes[idx], color=['#34495e', '#27ae60'])
        axes[idx].set_title(f"Performance: {m}")
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"comparison_{args.lang or 'all'}.png")
    print(f"\nDone. Results saved to results_{args.lang or 'all'}.csv")

if __name__ == "__main__":
    execute_research()
