import os
import time
import torch
import warnings
import numpy as np
import difflib
import matplotlib
import matplotlib.pyplot as plt
import re
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util

# --- Cluster Setup ---
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Load Token from environment
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Configuration
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
SEMANTIC_MODEL_ID = "all-MiniLM-L6-v2"

MAX_ITERATIONS = 2
SEMANTIC_CONVERGENCE = 0.95 

# -----------------------------
# 1. Semantic Auditor
# -----------------------------
class SemanticAuditor:
    def __init__(self):
        self.model = SentenceTransformer(SEMANTIC_MODEL_ID)

    def get_score(self, text1, text2):
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

# -----------------------------
# 2. Main Processor
# -----------------------------
class EchoRefineProcessor:
    def __init__(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading {LLAMA_MODEL_ID}...")
        self.llama_tok = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, token=HUGGING_FACE_HUB_TOKEN)
        self.llama_mod = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_ID, quantization_config=bnb_config, device_map="auto", token=HUGGING_FACE_HUB_TOKEN
        )
        
        print(f"Loading mBART-50...")
        self.nmt_tok = AutoTokenizer.from_pretrained(MBART_MODEL_ID)
        self.nmt_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_MODEL_ID, torch_dtype=torch.float16, device_map="auto")

    def mbart_translate(self, text, src_iso, tgt_iso):
        mapping = {"eng": "en_XX", "npi": "ne_NP"}
        self.nmt_tok.src_lang = mapping[src_iso]
        inputs = self.nmt_tok(text, return_tensors="pt", padding=True).to(self.nmt_mod.device)
        outputs = self.nmt_mod.generate(**inputs, forced_bos_token_id=self.nmt_tok.lang_code_to_id[mapping[tgt_iso]])
        return self.nmt_tok.decode(outputs[0], skip_special_tokens=True)

    def llama_direct(self, source):
        messages = [{"role": "user", "content": f"Translate to Nepali. Output only the Nepali text: {source}"}]
        prompt = self.llama_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        outputs = self.llama_mod.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.llama_tok.eos_token_id)
        return self.llama_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def llama_reason_and_refine(self, source, draft, back_trans):
        messages = [
            {"role": "system", "content": "You are a professional Nepali editor. Fix errors based on back-translation. Output the reasoning, then the final Nepali text after the marker 'RESULT:'"},
            {"role": "user", "content": f"Original English: {source}\nNepali Draft: {draft}\nBack-translation: {back_trans}\n\nAnalysis: Compare Original and Back-translation.\nRESULT: [Corrected Nepali]"}
        ]
        prompt = self.llama_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        outputs = self.llama_mod.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=self.llama_tok.eos_token_id)
        raw_res = self.llama_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        # Robust Extraction
        if "RESULT:" in raw_res:
            final_text = raw_res.split("RESULT:")[-1].strip()
        else:
            final_text = raw_res.split("\n")[-1].strip() # Fallback to last line

        # Language Guard: If output contains more than 20% English characters, it's failed. Revert to draft.
        english_chars = len(re.findall(r'[a-zA-Z]', final_text))
        if len(final_text) > 0 and (english_chars / len(final_text)) > 0.2:
            return draft
            
        return final_text

# -----------------------------
# 3. Execution Logic
# -----------------------------
def run_eval(num_samples=50):
    proc = EchoRefineProcessor()
    auditor = SemanticAuditor()
    
    metrics = {
        "chrf": evaluate.load("chrf"),
        "bleu": evaluate.load("bleu"),
        "bertscore": evaluate.load("bertscore"),
        "comet": evaluate.load("comet")
    }

    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    eng_df = df[df['iso_639_3'] == 'eng'].reset_index()
    npi_df = df[df['iso_639_3'] == 'npi'].reset_index()

    eval_store = {"mBART": [], "Llama_Direct": [], "EchoRefine": [], "Ref": [], "Src": []}

    for i in range(num_samples):
        src = eng_df.iloc[i]['text']
        ref = npi_df.iloc[i]['text']

        # Step A: Baselines
        mbart_draft = proc.mbart_translate(src, "eng", "npi")
        llama_dir = proc.llama_direct(src)
        
        # Step B: Iterative Refinement
        current_refined = mbart_draft
        for iteration in range(MAX_ITERATIONS):
            back = proc.mbart_translate(current_refined, "npi", "eng")
            score = auditor.get_score(src, back)
            if score >= SEMANTIC_CONVERGENCE: break
            current_refined = proc.llama_reason_and_refine(src, current_refined, back)

        eval_store["mBART"].append(mbart_draft)
        eval_store["Llama_Direct"].append(llama_dir)
        eval_store["EchoRefine"].append(current_refined)
        eval_store["Ref"].append(ref)
        eval_store["Src"].append(src)
        
        # DEBUG PRINTS INSIDE LOOP
        print(f"--- Sample {i} ---")
        print(f"Source: {src[:50]}...")
        print(f"Result: {current_refined[:50]}...")

    # Scoring
    final_scores = {}
    for model in ["mBART", "Llama_Direct", "EchoRefine"]:
        preds = eval_store[model]
        refs_nested = [[r] for r in eval_store["Ref"]]
        refs_flat = eval_store["Ref"]
        srcs = eval_store["Src"]

        ch = metrics["chrf"].compute(predictions=preds, references=refs_nested)['score']
        bl = metrics["bleu"].compute(predictions=preds, references=refs_nested)['bleu'] * 100
        bs = np.mean(metrics["bertscore"].compute(predictions=preds, references=refs_flat, lang="ne")['f1']) * 100
        cm = metrics["comet"].compute(predictions=preds, references=refs_flat, sources=srcs)['mean_score'] * 100
        final_scores[model] = [ch, bl, bs, cm]

    # Plot
    labels = ["chrF", "BLEU", "BERTScore", "COMET"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, final_scores["mBART"], width, label='mBART Baseline')
    ax.bar(x, final_scores["Llama_Direct"], width, label='Llama-3.1 Direct')
    ax.bar(x + width, final_scores["EchoRefine"], width, label='EchoRefine (BTI)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('echorefine_results.png')
    print("Done. Results saved to echorefine_results.png")

if __name__ == "__main__":
    run_eval(num_samples=5)
