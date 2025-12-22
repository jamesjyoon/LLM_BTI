import os
import time
import torch
import warnings
import numpy as np
import difflib
import matplotlib.pyplot as plt
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# 1. Configuration & Hardware
# -----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

AYA_MODEL_ID = "CohereForAI/aya-101" # 13B Parameters
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"

MAX_ITERATIONS = 3
LCS_CONVERGENCE_THRESHOLD = 0.92  # If similarity is this high, stop iterating

# -----------------------------
# 2. Similarity & Metrics
# -----------------------------
class LCSCalculator:
    @staticmethod
    def calculate(text1, text2):
        """Word-level Longest Common Subsequence ratio."""
        t1, t2 = text1.split(), text2.split()
        if not t1 or not t2: return 0.0
        return difflib.SequenceMatcher(None, t1, t2).ratio()

print("Loading Evaluation Metrics...")
chrf_metric = evaluate.load("chrf")
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")
comet_metric = evaluate.load("comet")

# -----------------------------
# 3. Local Model System
# -----------------------------
class IterativeBTISystem:
    def __init__(self):
        print("Loading Aya-101 and mBART-50 into Local VRAM...")
        # Aya-101 (Expert Refiner)
        self.aya_tok = AutoTokenizer.from_pretrained(AYA_MODEL_ID)
        self.aya_mod = AutoModelForSeq2SeqLM.from_pretrained(
            AYA_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
        
        # mBART-50 (NMT Drafter)
        self.nmt_tok = AutoTokenizer.from_pretrained(MBART_MODEL_ID)
        self.nmt_mod = AutoModelForSeq2SeqLM.from_pretrained(
            MBART_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )

    def mbart_translate(self, text, src_iso, tgt_iso):
        mapping = {"eng": "en_XX", "npi": "ne_NP"}
        self.nmt_tok.src_lang = mapping[src_iso]
        inputs = self.nmt_tok(text, return_tensors="pt", padding=True).to(self.nmt_mod.device)
        outputs = self.nmt_mod.generate(
            **inputs, 
            forced_bos_token_id=self.nmt_tok.lang_code_to_id[mapping[tgt_iso]],
            max_new_tokens=256
        )
        return self.nmt_tok.decode(outputs[0], skip_special_tokens=True)

    def aya_direct_translate(self, source):
        prompt = f"Translate the following English text into Nepali: {source}"
        inputs = self.aya_tok(prompt, return_tensors="pt").to(self.aya_mod.device)
        outputs = self.aya_mod.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.aya_tok.decode(outputs[0], skip_special_tokens=True).strip()

    def aya_refine(self, source, draft, back_trans):
        # Few-Shot Instruction Prompt
        prompt = (
            "Task: Correct the Nepali translation based on the back-translation errors.\n\n"
            "Example:\n"
            "English: The house is red.\n"
            "Draft Nepali: घर नीलो छ।\n"
            "Back-translation: The house is blue.\n"
            "Fixed Nepali: घर रातो छ।\n\n"
            f"Current Task:\n"
            f"English: {source}\n"
            f"Draft Nepali: {draft}\n"
            f"Back-translation: {back_trans}\n"
            "Fixed Nepali (Only the text):"
        )
        inputs = self.aya_tok(prompt, return_tensors="pt").to(self.aya_mod.device)
        outputs = self.aya_mod.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.aya_tok.decode(outputs[0], skip_special_tokens=True).strip()

# -----------------------------
# 4. The Iteration Engine
# -----------------------------
def run_bti_loop(system, lcs, source_text):
    # Step 1: Initial mBART Draft
    current_translation = system.mbart_translate(source_text, "eng", "npi")
    
    for iteration in range(MAX_ITERATIONS):
        # Step 2: Back-translate to check for errors
        back_trans = system.mbart_translate(current_translation, "npi", "eng")
        
        # Step 3: Check structural similarity
        score = lcs.calculate(source_text, back_trans)
        
        if score >= LCS_CONVERGENCE_THRESHOLD:
            # We have achieved high fidelity
            break
            
        # Step 4: Refine with LLM (Aya)
        current_translation = system.aya_refine(source_text, current_translation, back_trans)
        
    return current_translation

# -----------------------------
# 5. Main Evaluation
# -----------------------------
def execute_experiment(num_samples=50):
    system = IterativeBTISystem()
    lcs = LCSCalculator()
    
    print("Loading Flores dataset...")
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    eng_df = df[df['iso_639_3'] == 'eng'].reset_index()
    npi_df = df[df['iso_639_3'] == 'npi'].reset_index()

    # Data collection for scoring
    eval_data = {"mBART": [], "Aya_Direct": [], "LLM_BTI": [], "Ref": [], "Src": []}

    print(f"Running Evaluation on {num_samples} samples...")

    for i in range(num_samples):
        source = eng_df.iloc[i]['text']
        reference = npi_df.iloc[i]['text']

        # 1. mBART Baseline
        res_mbart = system.mbart_translate(source, "eng", "npi")
        
        # 2. Aya Direct Baseline
        res_aya = system.aya_direct_translate(source)
        
        # 3. Iterative BTI Pipeline
        res_bti = run_bti_loop(system, lcs, source)

        eval_data["mBART"].append(res_mbart)
        eval_data["Aya_Direct"].append(res_aya)
        eval_data["LLM_BTI"].append(res_bti)
        eval_data["Ref"].append(reference)
        eval_data["Src"].append(source)

        if (i+1) % 5 == 0:
            print(f"Sample {i+1}/{num_samples} done.")

    # -----------------------------
    # 6. Scoring & Metrics
    # -----------------------------
    metrics_summary = {}
    for model in ["mBART", "Aya_Direct", "LLM_BTI"]:
        preds = eval_data[model]
        refs = [[r] for r in eval_data["Ref"]]
        refs_flat = eval_data["Ref"]
        srcs = eval_data["Src"]

        print(f"Computing metrics for {model}...")
        c_score = chrf_metric.compute(predictions=preds, references=refs)['score']
        b_score = bleu_metric.compute(predictions=preds, references=refs)['bleu'] * 100
        bs_res = bertscore_metric.compute(predictions=preds, references=refs_flat, lang="ne")['f1']
        bs_avg = np.mean(bs_res) * 100
        cm_res = comet_metric.compute(predictions=preds, references=refs_flat, sources=srcs)['mean_score'] * 100

        metrics_summary[model] = [c_score, b_score, bs_avg, cm_res]

    # -----------------------------
    # 7. Visualization
    # -----------------------------
    labels = ["chrF", "BLEU", "BERTScore", "COMET"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, metrics_summary["mBART"], width, label='mBART (Baseline)')
    ax.bar(x, metrics_summary["Aya_Direct"], width, label='Aya (Direct LLM)')
    ax.bar(x + width, metrics_summary["LLM_BTI"], width, label='LLM-BTI (Iterative)')

    ax.set_ylabel('Scores')
    ax.set_title(f'Translation Comparison (English -> Nepali, N={num_samples})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig('local_bti_comparison.png')
    print("\nEvaluation Complete. Results saved to 'local_bti_comparison.png'")

if __name__ == "__main__":
    execute_experiment(num_samples=50)
