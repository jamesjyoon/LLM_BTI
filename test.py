import os
import time
import torch
import logging
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM, 
    pipeline
)
from datasets import load_dataset
import evaluate
from sentence_transformers import SentenceTransformer, util

# Set backend for matplotlib
matplotlib.use('Agg')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------
# Aya-Expanse-8B is very fast. For your 80GB VRAM, you could also use "CohereForAI/aya-23b"
LLM_MODEL_ID = "CohereForAI/aya-expanse-8b" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
SIMILARITY_MODEL = "all-MiniLM-L6-v2"

# -----------------------------
# Language Code Mapper (mBART)
# -----------------------------
def get_mbart_code(language: str) -> str:
    mapping = {
        "english": "en_XX", "nepali": "ne_NP", "japanese": "ja_XX",
        "hindi": "hi_IN", "thai": "th_TH", "vietnamese": "vi_VN",
        "swahili": "sw_KE", "french": "fr_XX", "spanish": "es_XX"
    }
    # Handle FLORES style inputs (e.g., eng_Latn -> en_XX)
    clean_lang = language.split('_')[0].lower()
    if clean_lang == "eng": return "en_XX"
    if clean_lang == "npi": return "ne_NP"
    return mapping.get(clean_lang, mapping.get(language.lower(), "en_XX"))

# -----------------------------
# Local LLM Handler (Aya)
# -----------------------------
class LocalAyaHandler:
    def __init__(self, model_id):
        print(f"Loading {model_id} onto local GPUs...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto", # Automatically balances across your 2x A100s
        )
        self.total_tokens = 0
        self.total_duration = 0

    def generate(self, prompt: str, max_new_tokens=512):
        start_time = time.perf_counter()
        
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response_ids = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        duration = time.perf_counter() - start_time
        self.total_duration += duration
        self.total_tokens += (len(input_ids[0]) + len(response_ids))
        
        return response, duration

# -----------------------------
# Translator Classes
# -----------------------------
class MBARTTranslator:
    def __init__(self, model_name=MBART_MODEL_ID):
        self.name = "mBART"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:0") # Put mBART on first GPU
        self.total_duration = 0
        self.total_tokens = 0

    def translate(self, source_text, src_lang, tgt_lang):
        start_time = time.perf_counter()
        src_code = get_mbart_code(src_lang)
        tgt_code = get_mbart_code(tgt_lang)

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(source_text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
            max_length=512
        )
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        self.total_duration += (time.perf_counter() - start_time)
        self.total_tokens += (inputs.input_ids.numel() + len(outputs[0]))
        return translation

class AyaTranslator:
    """Uses Aya for both direct translation and as a critique agent."""
    def __init__(self, aya_handler):
        self.name = "Aya"
        self.aya = aya_handler

    def translate(self, source_text, src_lang, tgt_lang):
        prompt = (
            f"Translate the following {src_lang} text to {tgt_lang}. "
            f"Provide ONLY the translation, no explanation.\n\n"
            f"Text: {source_text}"
        )
        translation, _ = self.aya.generate(prompt)
        return translation

    def refine(self, translation, source_text, back_translation, discrepancies, src_lang, tgt_lang):
        disc_str = "\n".join([f"- Original: {d['original']} | Back-Trans: {d['back_translation']}" for d in discrepancies])
        prompt = (
            f"Refine this {tgt_lang} translation of a {src_lang} text. "
            f"Discrepancies were found during back-translation:\n{disc_str}\n\n"
            f"Original Source: {source_text}\n"
            f"Current Translation: {translation}\n"
            f"Improved Translation (ONLY the text):"
        )
        improved, _ = self.aya.generate(prompt)
        return improved

# -----------------------------
# Similarity & Logic
# -----------------------------
class SentenceTransformerSimilarityCalculator:
    def __init__(self):
        self.model = SentenceTransformer(SIMILARITY_MODEL)

    def calculate_similarity(self, text1, text2):
        if not text1 or not text2: return 0.0
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()
    
    def extract_discrepancies(self, original, back_translation, threshold=0.75):
        # Simplified sentence split for demo
        orig_sents = [s.strip() for s in original.split('.') if s.strip()]
        back_sents = [s.strip() for s in back_translation.split('.') if s.strip()]
        if not orig_sents or not back_sents: return []
        
        discrepancies = []
        # Find lowest similarity matches
        for osent in orig_sents:
            sims = [self.calculate_similarity(osent, bsent) for bsent in back_sents]
            max_sim = max(sims) if sims else 0
            if max_sim < threshold:
                discrepancies.append({"original": osent, "back_translation": "Mismatch", "similarity": max_sim})
        return discrepancies

# -----------------------------
# Improvement Pipeline
# -----------------------------
class LLMBTI_Pipeline:
    def __init__(self, nmt, aya_translator, similarity_calc):
        self.nmt = nmt
        self.aya = aya_translator
        self.sim_calc = similarity_calc

    def run(self, source_text, src_lang, tgt_lang, max_iters=3):
        # 1. Base NMT Forward
        current_translation = self.nmt.translate(source_text, src_lang, tgt_lang)
        
        for i in range(max_iters):
            # 2. Back Translate
            back_translation = self.nmt.translate(current_translation, tgt_lang, src_lang)
            
            # 3. Check Similarity
            sim = self.sim_calc.calculate_similarity(source_text, back_translation)
            if sim > 0.92: break
            
            # 4. Extract Discrepancies
            discs = self.sim_calc.extract_discrepancies(source_text, back_translation)
            
            # 5. Refine with Aya
            current_translation = self.aya.refine(current_translation, source_text, back_translation, discs, src_lang, tgt_lang)
            
        return current_translation

# -----------------------------
# Evaluator
# -----------------------------
class FloresEvaluator:
    def __init__(self):
        self.dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
        self.df = self.dataset.to_pandas()
        self.comet = evaluate.load("comet")
        self.chrf = evaluate.load("chrf")
        self.bleu = evaluate.load("bleu")

    def evaluate_pair(self, src_code, tgt_code, num_examples, nmt, aya_trans, pipeline):
        src_lang, src_script = src_code.split("_")
        tgt_lang, tgt_script = tgt_code.split("_")
        
        subset = self.df[(self.df['iso_639_3'] == src_lang) & (self.df['iso_15924'] == src_script)].head(num_examples)
        tgt_subset = self.df[(self.df['iso_639_3'] == tgt_lang) & (self.df['iso_15924'] == tgt_script)].head(num_examples)
        
        results = {"nmt": [], "aya": [], "bti": []}
        
        for idx in range(len(subset)):
            source = subset.iloc[idx]['text']
            ref = tgt_subset.iloc[idx]['text']
            
            # Run Methods
            res_nmt = nmt.translate(source, src_lang, tgt_lang)
            res_aya = aya_trans.translate(source, src_lang, tgt_lang)
            res_bti = pipeline.run(source, src_lang, tgt_lang)
            
            # Scoring
            results["nmt"].append(self.chrf.compute(predictions=[res_nmt], references=[[ref]])['score'])
            results["aya"].append(self.chrf.compute(predictions=[res_aya], references=[[ref]])['score'])
            results["bti"].append(self.chrf.compute(predictions=[res_bti], references=[[ref]])['score'])
            
            print(f"Example {idx+1}/{num_examples} processed.")

        print(f"\nAverage chrF Scores for {src_code} -> {tgt_code}:")
        print(f"mBART Base: {np.mean(results['nmt']):.2f}")
        print(f"Aya Direct: {np.mean(results['aya']):.2f}")
        print(f"LLM-BTI (mBART+Aya): {np.mean(results['bti']):.2f}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Setup
    aya_handler = LocalAyaHandler(LLM_MODEL_ID)
    aya_trans = AyaTranslator(aya_handler)
    nmt = MBARTTranslator()
    sim_calc = SentenceTransformerSimilarityCalculator()
    bti_pipeline = LLMBTI_Pipeline(nmt, aya_trans, sim_calc)
    
    evaluator = FloresEvaluator()
    
    # Run Eval
    evaluator.evaluate_pair(
        src_code="eng_Latn", 
        tgt_code="npi_Deva", 
        num_examples=5, 
        nmt=nmt, 
        aya_trans=aya_trans, 
        pipeline=bti_pipeline
    )
