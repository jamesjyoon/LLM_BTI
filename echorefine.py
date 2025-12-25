import os
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from scipy.stats import bootstrap
from dotenv import load_dotenv
from datasets import load_dataset
import evaluate
import sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from comet import download_model, load_from_checkpoint

# --- TOP OF SCRIPT ---
os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
import datasets
import logging

transformers.utils.logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_error()
datasets.utils.logging.disable_progress_bar()
datasets.disable_progress_bar()

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg') 

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
    # Add more here as needed
]

# -----------------------------
# 2. System Architecture
# -----------------------------
class EchoRefineSystem:
    def __init__(self):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        print("Loading Llama-3.1-70B...")
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID)
        self.l_mod = AutoModelForCausalLM.from_pretrained(LLAMA_ID, quantization_config=bnb, device_map="auto")
        
        print("Loading mBART-50...")
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, torch_dtype=torch.float16, device_map="auto")

        print("Loading CometKiwi Judge...")
        qe_path = download_model(INTERNAL_QE_ID)
        self.qe_mod = load_from_checkpoint(qe_path).to("cuda")

    def mbart_translate(self, text, src_tag, tgt_tag):
        self.n_tok.src_lang = src_tag
        inputs = self.n_tok(text, return_tensors="pt", truncation=True).to(self.n_mod.device)
        outputs = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt_tag])
        return self.n_tok.decode(outputs[0], skip_special_tokens=True)

    def refine_with_cot(self, source, draft, back_trans, lang_name):
        """Improved Prompt focusing on Back-translation reconciliation."""
        messages = [
            {"role": "system", "content": f"You are a linguistic expert. Your task is to fix the '{lang_name} Draft' so its meaning matches the 'Original English'. Use the 'Back-translation' to find where the draft went wrong."},
            {"role": "user", "content": (
                f"Original English: {source}\n"
                f"Current {lang_name} Draft: {draft}\n"
                f"English Back-translation of Draft: {back_trans}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Compare the Original English to the Back-translation.\n"
                f"2. If they differ, rewrite the {lang_name} Draft to fix the error.\n"
                f"3. Do not add conversational filler. Provide only the text.\n\n"
                f"RESULT: [Fixed {lang_name} text only]"
            )}
        ]
        prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.l_mod.device)
        outputs = self.l_mod.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=self.l_tok.eos_token_id)
        res = self.l_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        final = res.split("RESULT:")[-1].strip() if "RESULT:" in res else res
        if len(re.findall(r'[a-zA-Z]', final)) > (len(final) * 0.3): return draft
        return final

    def judge(self, source, draft, refined):
        """Soft-selection logic with a grace margin for the LLM."""
        data = [{"src": source, "mt": draft}, {"src": source, "mt": refined}]
        with torch.no_grad():
            scores = self.qe_mod.predict(data, batch_size=2, gpus=1, progress_bar=False).scores
        
        # Marginal logic: Accept LLM if it's better or only slightly worse (to favor fluency)
        if scores[1] > (scores[0] - 0.02): 
            return refined, "LLM"
        return draft, "mBART"

# -----------------------------
# 3. Stats & Main Loop
# -----------------------------
def execute_research():
    sys = EchoRefineSystem()
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    
    chrf = evaluate.load("chrf")
    comet_ref = evaluate.load("comet", FINAL_COMET_ID)

    final_report = []

    for cfg in EXPERIMENT_CONFIG:
        print(f"\nEvaluating Tier: {cfg['tier']} | Language: {cfg['name']}")
        
        src_texts = df[df['iso_639_3'] == 'eng']['text'].tolist()
        ref_texts = df[df['iso_639_3'] == cfg['iso']]['text'].tolist()
        
        # Use full devtest (1012) for paper; using 10 for test
        n = min(len(src_texts), 50) 
        
        storage = {"mBART": [], "EchoRefine": []}
        selection_counts = {"LLM": 0, "mBART": 0}
        
        for i in tqdm(range(n), disable=False):
            source = src_texts[i]
            # 1. Draft & Back-trans
            draft = sys.mbart_translate(source, "en_XX", cfg['mbart'])
            back = sys.mbart_translate(draft, cfg['mbart'], "en_XX")
            # 2. Refine & Judge
            refined = sys.refine_with_cot(source, draft, back, cfg['name'])
            final, winner = sys.judge(source, draft, refined)

            storage["mBART"].append(draft)
            storage["EchoRefine"].append(final)
            selection_counts[winner] += 1

        print(f"Selection Rate for {cfg['name']}: {selection_counts}")

        # Metrics
        row = {"Language": cfg['name'], "Tier": cfg['tier']}
        for key in ["mBART", "EchoRefine"]:
            preds = storage[key]
            refs = [[r] for r in ref_texts[:n]]
            
            row[f"{key}_BLEU"] = sacrebleu.corpus_bleu(preds, refs).score
            row[f"{key}_chrF"] = chrf.compute(predictions=preds, references=refs)['score']
            row[f"{key}_COMET"] = comet_ref.compute(predictions=preds, references=ref_texts[:n], sources=src_texts[:n])['mean_score'] * 100
        
        final_report.append(row)

    # -----------------------------
    # 4. Comprehensive Graphing
    # -----------------------------
    results_df = pd.DataFrame(final_report)
    results_df.to_csv("echorefine_paper_results.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['BLEU', 'chrF', 'COMET']
    colors = ['#3498db', '#2ecc71']

    for i, metric in enumerate(metrics):
        cols = [f"mBART_{metric}", f"EchoRefine_{metric}"]
        results_df.plot(x="Language", y=cols, kind="bar", ax=axes[i], color=colors, legend=(i==0))
        axes[i].set_title(f"Comparison: {metric}")
        axes[i].set_ylabel("Score")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("paper_full_metrics_comparison.png")
    print("\nBenchmark complete. Summary saved to echorefine_paper_results.csv")

if __name__ == "__main__":
    execute_research()
