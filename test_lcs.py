import os
import time
import torch
import numpy as np
import difflib
import matplotlib.pyplot as plt
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Configuration
# -----------------------------
AYA_MODEL_ID = "CohereForAI/aya-101"
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
LCS_THRESHOLD = 0.85  # Only refine if similarity is lower than this

# -----------------------------
# LCS Similarity Calculator
# -----------------------------
class LCSCalculator:
    @staticmethod
    def calculate(text1, text2):
        """Word-level Longest Common Subsequence ratio."""
        t1, t2 = text1.split(), text2.split()
        if not t1 or not t2: return 0.0
        return difflib.SequenceMatcher(None, t1, t2).ratio()

# -----------------------------
# Local Model System
# -----------------------------
class OptimizedBTIProcessor:
    def __init__(self):
        print("Loading local weights to VRAM...")
        # Load Aya-101 (13B) - Refiner
        self.aya_tok = AutoTokenizer.from_pretrained(AYA_MODEL_ID)
        self.aya_mod = AutoModelForSeq2SeqLM.from_pretrained(
            AYA_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
        
        # Load mBART-50 - Drafter
        self.nmt_tok = AutoTokenizer.from_pretrained(MBART_MODEL_ID)
        self.nmt_mod = AutoModelForSeq2SeqLM.from_pretrained(
            MBART_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )

    def mbart_translate(self, text, src_iso, tgt_iso):
        mapping = {"eng": "en_XX", "npi": "ne_NP"}
        self.nmt_tok.src_lang = mapping[src_iso]
        inputs = self.nmt_tok(text, return_tensors="pt").to(self.nmt_mod.device)
        outputs = self.nmt_mod.generate(
            **inputs, 
            forced_bos_token_id=self.nmt_tok.lang_code_to_id[mapping[tgt_iso]]
        )
        return self.nmt_tok.decode(outputs[0], skip_special_tokens=True)

    def aya_refine(self, source, draft, back_trans):
        # Few-Shot Prompting to guide the LLM
        few_shot_prompt = (
            "Task: Correct the Nepali translation based on the provided back-translation.\n\n"
            "Example 1:\n"
            "English: I am eating an apple.\n"
            "Draft Nepali: म स्याउ खाँदैछु।\n"
            "Back-translation: I am eating an apple.\n"
            "Fixed Nepali: म स्याउ खाँदैछु।\n\n"
            "Example 2:\n"
            "English: The weather is very cold today.\n"
            "Draft Nepali: आज मौसम तातो छ।\n"
            "Back-translation: The weather is hot today.\n"
            "Fixed Nepali: आज मौसम धेरै चिसो छ।\n\n"
            f"Current Task:\n"
            f"English: {source}\n"
            f"Draft Nepali: {draft}\n"
            f"Back-translation: {back_trans}\n"
            "Fixed Nepali (ONLY the text):"
        )
        
        inputs = self.aya_tok(few_shot_prompt, return_tensors="pt").to(self.aya_mod.device)
        # Deterministic generation for faithfulness
        outputs = self.aya_mod.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.aya_tok.decode(outputs[0], skip_special_tokens=True).strip()

# -----------------------------
# Main Evaluation
# -----------------------------
def run_optimized_experiment(num_samples=5):
    proc = OptimizedBTIProcessor()
    lcs = LCSCalculator()
    
    # Load Metrics
    chrf = evaluate.load("chrf")
    bleu = evaluate.load("bleu")
    
    # Load Data (Local Cache)
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    eng_df = df[df['iso_639_3'] == 'eng'].reset_index()
    npi_df = df[df['iso_639_3'] == 'npi'].reset_index()

    results = {"mBART": [], "BTI_Optimized": []}
    counts = {"Refined": 0, "Kept_Original": 0}

    print(f"Starting optimized evaluation on {num_samples} samples...")

    for i in range(num_samples):
        source = eng_df.iloc[i]['text']
        reference = npi_df.iloc[i]['text']

        # 1. Draft with mBART
        draft = proc.mbart_translate(source, "eng", "npi")
        
        # 2. Back-translate
        back = proc.mbart_translate(draft, "npi", "eng")
        
        # 3. Trigger Logic: Check LCS similarity between Original and Back-translation
        sim_score = lcs.calculate(source, back)
        
        if sim_score < LCS_THRESHOLD:
            # Improvement needed
            final_translation = proc.aya_refine(source, draft, back)
            counts["Refined"] += 1
        else:
            # Draft is likely already excellent
            final_translation = draft
            counts["Kept_Original"] += 1

        # Score
        results["mBART"].append(chrf.compute(predictions=[draft], references=[[reference]])['score'])
        results["BTI_Optimized"].append(chrf.compute(predictions=[final_translation], references=[[reference]])['score'])

        if (i+1) % 5 == 0:
            print(f"Progress: {i+1}/{num_samples} | LCS trigger rate: {counts['Refined']/(i+1):.2%}")

    # Final Stats
    avg_mbart = np.mean(results["mBART"])
    avg_bti = np.mean(results["BTI_Optimized"])
    
    print("\n" + "="*40)
    print(f"RESULTS (Average chrF)")
    print(f"mBART Base      : {avg_mbart:.2f}")
    print(f"BTI (Optimized) : {avg_bti:.2f}")
    print(f"Improvement     : {avg_bti - avg_mbart:+.2f}")
    print(f"Refinement Rate : {counts['Refined']/num_samples:.1%}")
    print("="*40)

    # Plot
    plt.bar(["mBART Base", "BTI (Optimized)"], [avg_mbart, avg_bti], color=['gray', 'blue'])
    plt.ylabel("chrF Score")
    plt.title("Optimized LLM-BTI Performance (English-Nepali)")
    plt.savefig("optimized_bti_results.png")

if __name__ == "__main__":
    run_optimized_experiment()
