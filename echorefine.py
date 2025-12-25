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
import os
import warnings
import logging

# Set environment variables BEFORE importing heavy libraries
os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
import datasets
import torch

# Disable progress bars in the libraries
transformers.utils.logging.disable_progress_bar()
datasets.utils.logging.disable_progress_bar()

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ... rest of your imports and code ...
# --- Cluster & Environment Setup ---
import matplotlib
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# -----------------------------
# 1. Configuration & Tiers
# -----------------------------
LLAMA_ID = "meta-llama/Llama-3.1-70B-Instruct" 
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
INTERNAL_QE_ID = "Unbabel/wmt22-cometkiwi-da"
FINAL_COMET_ID = "Unbabel/wmt22-comet-da"

# Tiered Language Strategy
EXPERIMENT_CONFIG = [
    {"iso": "ben", "name": "Bengali", "mbart": "bn_IN", "tier": "Mid-Res"},
    {"iso": "sin", "name": "Sinhala", "mbart": "si_LK", "tier": "Low-Res"},
    {"iso": "npi", "name": "Nepali", "mbart": "ne_NP", "tier": "Low-Res"},
    {"iso": "mya", "name": "Burmese", "mbart": "my_MM", "tier": "Low-Res"},
    {"iso": "amh", "name": "Amharic", "mbart": "am_ET", "tier": "Low-Res"},
    {"iso": "hin", "name": "Hindi", "mbart": "hi_IN", "tier": "Mid-Res"},
    {"iso": "kor", "name": "Korean", "mbart": "ko_KR", "tier": "Qualitative"},
    {"iso": "fra", "name": "French", "mbart": "fr_XX", "tier": "High-Res"}
]

# -----------------------------
# 2. System Architecture
# -----------------------------
class EchoRefineSystem:
    def __init__(self):
        # 4-bit config for Llama-70B
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        print("Initializing LLM Reasoning Engine (Llama-3.1-70B)...")
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID)
        self.l_mod = AutoModelForCausalLM.from_pretrained(LLAMA_ID, quantization_config=bnb, device_map="auto")
        
        print("Initializing NMT Drafting Engine (mBART-50)...")
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

        print("Initializing Internal QE Judge (CometKiwi)...")
        qe_path = download_model(INTERNAL_QE_ID)
        self.qe_mod = load_from_checkpoint(qe_path).to("cuda")

    def mbart_translate(self, text, src_tag, tgt_tag):
        self.n_tok.src_lang = src_tag
        inputs = self.n_tok(text, return_tensors="pt", truncation=True).to(self.n_mod.device)
        outputs = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt_tag])
        return self.n_tok.decode(outputs[0], skip_special_tokens=True)

    def refine_with_cot(self, source, draft, back_trans, lang_name):
        """Chain-of-Thought refinement including error typology."""
        messages = [
            {"role": "system", "content": f"You are a linguistic expert in {lang_name}. Analyze translation errors and fix them."},
            {"role": "user", "content": (
                f"Source (EN): {source}\nDraft ({lang_name}): {draft}\nBack-translation (EN): {back_trans}\n\n"
                f"Identify if there are omissions, particle errors, or politeness mismatches. "
                f"Then provide the corrected {lang_name} translation after 'RESULT:'"
            )}
        ]
        prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.l_mod.device)
        outputs = self.l_mod.generate(**inputs, max_new_tokens=300, do_sample=False)
        res = self.l_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        final = res.split("RESULT:")[-1].strip() if "RESULT:" in res else res
        # Language Guard: ensure we don't return English reasoning
        if len(re.findall(r'[a-zA-Z]', final)) > (len(final) * 0.3): return draft
        return final

    def judge(self, source, draft, refined):
        data = [{"src": source, "mt": draft}, {"src": source, "mt": refined}]
        with torch.no_grad():
            scores = self.qe_mod.predict(data, batch_size=2, gpus=1).scores
        return refined if scores[1] > scores[0] else draft

# -----------------------------
# 3. Statistical Tools
# -----------------------------
def compute_p_value(refs, sys1, sys2):
    """Paired Bootstrap Resampling for BLEU p-value."""
    def bleu_diff(idx):
        s1 = [sys1[i] for i in idx]
        s2 = [sys2[i] for i in idx]
        r = [[refs[i]] for i in idx]
        return sacrebleu.corpus_bleu(s2, r).score - sacrebleu.corpus_bleu(s1, r).score

    # Using 1000 iterations for significance
    data = (np.arange(len(refs)),)
    res = bootstrap(data, bleu_diff, n_resamples=1000, method='percentile')
    # If 0 is not in the confidence interval, it is likely significant
    is_significant = res.confidence_interval.low > 0
    return is_significant

# -----------------------------
# 4. Main Research Loop
# -----------------------------
def execute_research():
    sys = EchoRefineSystem()
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    
    # Metrics
    chrf = evaluate.load("chrf")
    bert = evaluate.load("bertscore")
    comet_ref = evaluate.load("comet", FINAL_COMET_ID)

    final_report = []

    for cfg in EXPERIMENT_CONFIG:
        print(f"\nEvaluating Tier: {cfg['tier']} | Language: {cfg['name']}")
        
        src_texts = df[df['iso_639_3'] == 'eng']['text'].tolist()
        ref_texts = df[df['iso_639_3'] == cfg['iso']]['text'].tolist()
        
        # Limit to full devtest (1012) or 500 for time efficiency
        n = min(len(src_texts), 5) 
        
        storage = {"mBART": [], "EchoRefine": []}
        
        for i in tqdm(range(n)):
            source = src_texts[i]
            reference = ref_texts[i]

            # 1. Draft
            draft = sys.mbart_translate(source, "en_XX", cfg['mbart'])
            # 2. Back-translate
            back = sys.mbart_translate(draft, cfg['mbart'], "en_XX")
            # 3. Refine
            refined = sys.refine_with_cot(source, draft, back, cfg['name'])
            # 4. Judge
            final = sys.judge(source, draft, refined)

            storage["mBART"].append(draft)
            storage["EchoRefine"].append(final)

        # --- Calculate Metrics ---
        row = {"Language": cfg['name'], "Tier": cfg['tier']}
        for key in ["mBART", "EchoRefine"]:
            preds = storage[key]
            refs = [[r] for r in ref_texts[:n]]
            
            row[f"{key}_BLEU"] = sacrebleu.corpus_bleu(preds, refs).score
            row[f"{key}_chrF"] = chrf.compute(predictions=preds, references=refs)['score']
            row[f"{key}_COMET"] = comet_ref.compute(predictions=preds, references=ref_texts[:n], sources=src_texts[:n])['mean_score'] * 100

        # Signficance Test
        row["Significant"] = compute_p_value(ref_texts[:n], storage["mBART"], storage["EchoRefine"])
        final_report.append(row)

    # --- Save Results ---
    results_df = pd.DataFrame(final_report)
    results_df.to_csv("echorefine_paper_results.csv", index=False)
    print("\nResults saved to echorefine_paper_results.csv")
    
    # Generate Plot
    results_df.plot(x="Language", y=["mBART_COMET", "EchoRefine_COMET"], kind="bar", figsize=(12,6))
    plt.title("EchoRefine vs mBART Baseline across Tiers")
    plt.savefig("paper_comet_comparison.png")

# def save_qualitative_sample(source, mbart_res, echo_res, back_trans, lang):
#     with open(f"qualitative_{lang}.txt", "a", encoding="utf-8") as f:
#         f.write(f"SOURCE: {source}\n")
#         f.write(f"mBART DRAFT: {mbart_res}\n")
#         f.write(f"BACK-TRANS: {back_trans}\n")
#         f.write(f"ECHOREFINE: {echo_res}\n")
#         f.write("-" * 30 + "\n")

if __name__ == "__main__":
    execute_research()
