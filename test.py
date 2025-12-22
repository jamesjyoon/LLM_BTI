import os
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
from sentence_transformers import SentenceTransformer, util

# Configuration
# Aya-101 supports 101 languages including Nepali
LLM_MODEL_ID = "CohereForAI/aya-101" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"

# -----------------------------
# Language Code Mapper (mBART)
# -----------------------------
def get_mbart_code(language: str) -> str:
    mapping = {"english": "en_XX", "nepali": "ne_NP", "hindi": "hi_IN"}
    clean_lang = language.split('_')[0].lower()
    if clean_lang == "eng": return "en_XX"
    if clean_lang == "npi": return "ne_NP"
    return mapping.get(clean_lang, "en_XX")

# -----------------------------
# Aya-101 Handler (Seq2Seq)
# -----------------------------
class LocalAya101Handler:
    def __init__(self, model_id):
        print(f"Loading Aya-101 (13B) onto local GPUs...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Aya-101 is ~26GB in float16, fits easily on one A100-40GB
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto" 
        )

    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False # Keep it deterministic for evaluation
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# -----------------------------
# mBART Handler
# -----------------------------
class MBARTTranslator:
    def __init__(self, model_name=MBART_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load mBART on the first GPU
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:0")

    def translate(self, source_text, src_lang, tgt_lang):
        src_code = get_mbart_code(src_lang)
        tgt_code = get_mbart_code(tgt_lang)
        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(source_text, return_tensors="pt", padding=True).to("cuda:0")
        
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
            max_length=256
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# -----------------------------
# Similarity Logic
# -----------------------------
class SimilarityCalculator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def calculate_similarity(self, text1, text2):
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

# -----------------------------
# Main Evaluation Pipeline
# -----------------------------
def run_evaluation():
    # 1. Init Local Models
    aya = LocalAya101Handler(LLM_MODEL_ID)
    nmt = MBARTTranslator()
    sim_calc = SimilarityCalculator()
    
    # 2. Load Dataset
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    chrf = evaluate.load("chrf")

    # Filter for English and Nepali
    eng_df = df[df['iso_639_3'] == 'eng'].reset_index()
    npi_df = df[df['iso_639_3'] == 'npi'].reset_index()

    num_samples = 50
    results = {"mBART_Only": [], "LLM_BTI": [], "Aya_Direct": []}

    print(f"Starting Local Evaluation (English -> Nepali)...")

    for i in range(num_samples):
        source = eng_df.iloc[i]['text']
        reference = npi_df.iloc[i]['text']

        # Method 1: mBART Direct
        mbart_translation = nmt.translate(source, "english", "nepali")

        # Method 2: Aya Direct
        aya_prompt = f"Translate from English to Nepali: {source}"
        aya_direct = aya.generate(aya_prompt)

        # Method 3: LLM-BTI (Refinement)
        # Back-translate with mBART
        back_trans = nmt.translate(mbart_translation, "nepali", "english")
        
        # Refine with Aya
        refine_prompt = (
            f"Original English: {source}\n"
            f"Draft Nepali Translation: {mbart_translation}\n"
            f"Back-translation to English: {back_trans}\n"
            f"Please provide an improved Nepali translation based on the errors found in the back-translation."
        )
        bti_translation = aya.generate(refine_prompt)

        # Score them
        results["mBART_Only"].append(chrf.compute(predictions=[mbart_translation], references=[[reference]])['score'])
        results["Aya_Direct"].append(chrf.compute(predictions=[aya_direct], references=[[reference]])['score'])
        results["LLM_BTI"].append(chrf.compute(predictions=[bti_translation], references=[[reference]])['score'])

        if i % 5 == 0:
            print(f"Progress: {i}/{num_samples}")

    print("\n" + "="*30)
    print(f"RESULTS (Avg chrF)")
    print(f"mBART Only: {np.mean(results['mBART_Only']):.2f}")
    print(f"Aya Direct: {np.mean(results['Aya_Direct']):.2f}")
    print(f"LLM-BTI (Refined): {np.mean(results['LLM_BTI']):.2f}")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()
