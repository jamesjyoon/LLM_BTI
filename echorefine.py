import os
import time
import torch
import warnings
import numpy as np
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

HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Configuration
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
SEMANTIC_MODEL_ID = "all-MiniLM-L6-v2"

MAX_ITERATIONS = 2
SEMANTIC_THRESHOLD = 0.90 # Start refinement if similarity is below this

# -----------------------------
# 1. Models & Loaders
# -----------------------------
class EchoRefineSystem:
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
        
        print(f"Loading mBART-50 and COMET Judge...")
        self.nmt_tok = AutoTokenizer.from_pretrained(MBART_MODEL_ID)
        self.nmt_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        
        self.sim_model = SentenceTransformer(SEMANTIC_MODEL_ID)
        self.comet_metric = evaluate.load("comet") # The Judge

    def mbart_translate(self, text, src_iso, tgt_iso):
        mapping = {"eng": "en_XX", "npi": "ne_NP"}
        self.nmt_tok.src_lang = mapping[src_iso]
        inputs = self.nmt_tok(text, return_tensors="pt", padding=True).to(self.nmt_mod.device)
        outputs = self.nmt_mod.generate(**inputs, forced_bos_token_id=self.nmt_tok.lang_code_to_id[mapping[tgt_iso]])
        return self.nmt_tok.decode(outputs[0], skip_special_tokens=True)

    def llama_refine(self, source, draft, back_trans):
        messages = [
            {"role": "system", "content": "You are a Nepali translation auditor. Fix errors concisely. Do not add explanations. Keep the length similar to the draft."},
            {"role": "user", "content": f"Original English: {source}\nNepali Draft: {draft}\nBack-translation: {back_trans}\n\nRESULT: [Corrected Nepali]"}
        ]
        prompt = self.llama_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        outputs = self.llama_mod.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.llama_tok.eos_token_id)
        res = self.llama_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        final = res.split("RESULT:")[-1].strip() if "RESULT:" in res else res
        # Language Guard
        if len(re.findall(r'[a-zA-Z]', final)) > (len(final) * 0.25): return draft
        return final

    def judge_and_select(self, source, original_draft, refined_version):
        """Uses COMET to ensure the refinement actually improved quality."""
        # Note: COMET doesn't need a reference for Quality Estimation (QE) if we use references, 
        # but here we use the metric's standard score for a fair comparison.
        scores = self.comet_metric.compute(
            sources=[source, source],
            predictions=[original_draft, refined_version],
            references=[original_draft, original_draft] # Placeholder as we are relative judging
        )['scores']
        
        return refined_version if scores[1] > scores[0] else original_draft

# -----------------------------
# 2. Main Evaluation
# -----------------------------
def run_experiment(num_samples=50):
    sys = EchoRefineSystem()
    chrf = evaluate.load("chrf"); bleu = evaluate.load("bleu"); bert = evaluate.load("bertscore")

    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    eng_df = df[df['iso_639_3'] == 'eng'].reset_index()
    npi_df = df[df['iso_639_3'] == 'npi'].reset_index()

    store = {"mBART": [], "EchoRefine": [], "Ref": [], "Src": []}

    print(f"Starting experiment on N={num_samples}...")

    for i in range(num_samples):
        src = eng_df.iloc[i]['text']
        ref = npi_df.iloc[i]['text']

        # 1. Draft
        draft = sys.mbart_translate(src, "eng", "npi")
        
        # 2. Back-translate & Check
        back = sys.mbart_translate(draft, "npi", "eng")
        emb1 = sys.sim_model.encode(src, convert_to_tensor=True)
        emb2 = sys.sim_model.encode(back, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2).item()

        # 3. Refine with Reranking
        if sim < SEMANTIC_THRESHOLD:
            candidate = sys.llama_refine(src, draft, back)
            # Internal Quality Check
            final_res = sys.judge_and_select(src, draft, candidate)
        else:
            final_res = draft

        store["mBART"].append(draft)
        store["EchoRefine"].append(final_res)
        store["Ref"].append(ref); store["Src"].append(src)
        
        print(f"Sample {i+1}/{num_samples} | Sim: {sim:.2f} | Result: {final_res[:40]}...")

    # -----------------------------
    # 3. Final Scoring & Plots
    # -----------------------------
    results = {}
    for key in ["mBART", "EchoRefine"]:
        preds = store[key]; refs = [[r] for r in store["Ref"]]
        
        c = chrf.compute(predictions=preds, references=refs)['score']
        b = bleu.compute(predictions=preds, references=refs)['bleu'] * 100
        bs = np.mean(bert.compute(predictions=preds, references=store["Ref"], lang="ne")['f1']) * 100
        cm = sys.comet_metric.compute(predictions=preds, references=store["Ref"], sources=store["Src"])['mean_score'] * 100
        results[key] = [c, b, bs, cm]

    # Plot
    labels = ["chrF", "BLEU", "BERTScore", "COMET"]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, results["mBART"], width, label='mBART Baseline', color='#3498db')
    plt.bar(x + width/2, results["EchoRefine"], width, label='EchoRefine (Reranked)', color='#2ecc71')
    plt.xticks(x, labels); plt.legend(); plt.ylabel("Score"); plt.title(f"Final Comparison (N={num_samples})")
    plt.savefig('final_echorefine_v2.png')
    
    print("\nFINAL SCORES:")
    for k, v in results.items():
        print(f"{k}: chrF={v[0]:.2f}, BLEU={v[1]:.2f}, BERT={v[2]:.2f}, COMET={v[3]:.2f}")

if __name__ == "__main__":
    run_experiment(num_samples=5)
