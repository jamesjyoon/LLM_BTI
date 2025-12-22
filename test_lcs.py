import os
import time
import torch
import numpy as np
import difflib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate

# Hardware Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local Model IDs
AYA_MODEL_ID = "CohereForAI/aya-101" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"

# -----------------------------
# LCS Similarity Calculator
# -----------------------------
class LCSWordSimilarityCalculator:
    def calculate_similarity(self, text1, text2):
        tokens1, tokens2 = text1.split(), text2.split()
        if not tokens1 or not tokens2: return 0.0
        return difflib.SequenceMatcher(None, tokens1, tokens2).ratio()

# -----------------------------
# Local Handlers
# -----------------------------
class LocalTranslationSystem:
    def __init__(self):
        print("Loading Models to Local VRAM...")
        self.aya_tok = AutoTokenizer.from_pretrained(AYA_MODEL_ID)
        self.aya_mod = AutoModelForSeq2SeqLM.from_pretrained(AYA_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        
        self.nmt_tok = AutoTokenizer.from_pretrained(MBART_MODEL_ID)
        self.nmt_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        
        self.lcs_calc = LCSWordSimilarityCalculator()

    def mbart_translate(self, text, src_iso, tgt_iso):
        # ISO Mapping for mBART
        mapping = {"eng": "en_XX", "npi": "ne_NP"}
        src, tgt = mapping.get(src_iso), mapping.get(tgt_iso)
        
        self.nmt_tok.src_lang = src
        inputs = self.nmt_tok(text, return_tensors="pt").to(self.nmt_mod.device)
        outputs = self.nmt_mod.generate(**inputs, forced_bos_token_id=self.nmt_tok.lang_code_to_id[tgt])
        return self.nmt_tok.decode(outputs[0], skip_special_tokens=True)

    def aya_refine(self, source, draft, back_trans):
        prompt = (
            f"Original: {source}\nDraft: {draft}\nBack-translation: {back_trans}\n"
            f"Fix the Draft translation to better match the Original English. Output only the translation."
        )
        inputs = self.aya_tok(prompt, return_tensors="pt").to(self.aya_mod.device)
        outputs = self.aya_mod.generate(**inputs, max_new_tokens=256)
        return self.aya_tok.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Run Evaluation
# -----------------------------
def run_test():
    system = LocalTranslationSystem()
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    chrf = evaluate.load("chrf")

    src_data = df[df['iso_639_3'] == 'eng'].reset_index()
    tgt_data = df[df['iso_639_3'] == 'npi'].reset_index()

    results = {"mbart": [], "bti_lcs": []}

    for i in range(5): # Test with 20 samples
        source = src_data.iloc[i]['text']
        ref = tgt_data.iloc[i]['text']

        # 1. Draft
        draft = system.mbart_translate(source, "eng", "npi")
        
        # 2. Back-translate
        back = system.mbart_translate(draft, "npi", "eng")
        
        # 3. Use LCS to decide if refinement is needed
        lcs_score = system.lcs_calc.calculate_similarity(source, back)
        
        if lcs_score < 0.85: # Threshold: If structural similarity is low, refine
            final_trans = system.aya_refine(source, draft, back)
        else:
            final_trans = draft

        results["mbart"].append(chrf.compute(predictions=[draft], references=[[ref]])['score'])
        results["bti_lcs"].append(chrf.compute(predictions=[final_trans], references=[[ref]])['score'])
        
        print(f"Sample {i} | LCS Score: {lcs_score:.2f} | chrF Diff: {results['bti_lcs'][-1] - results['mbart'][-1]:.2f}")

    print(f"\nAvg mBART chrF: {np.mean(results['mbart']):.2f}")
    print(f"Avg BTI (LCS-Triggered) chrF: {np.mean(results['bti_lcs']):.2f}")

if __name__ == "__main__":
    run_test()
