import os
import time
import torch
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util

# --- Cluster Setup ---
matplotlib.use('Agg') # Necessary for remote/cluster environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# --- Configuration ---
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
SEMANTIC_MODEL_ID = "all-MiniLM-L6-v2"
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

MAX_ITERATIONS = 2
SEMANTIC_CONVERGENCE = 0.95 

# -----------------------------
# 1. Advanced Verification Tools
# -----------------------------
class SemanticAuditor:
    def __init__(self):
        self.model = SentenceTransformer(SEMANTIC_MODEL_ID)

    def get_score(self, text1, text2):
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

# -----------------------------
# 2. Local Model Processor
# -----------------------------
class EchoRefineProcessor:
    def __init__(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading Llama-3.1-70B (4-bit)...")
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
        messages = [{"role": "user", "content": f"Translate to Nepali. Only output the text: {source}"}]
        prompt = self.llama_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        outputs = self.llama_mod.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.llama_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def llama_reason_and_refine(self, source, draft, back_trans):
        messages = [
            {"role": "system", "content": "Analyze English-to-English shifts to fix the Nepali Draft."},
            {"role": "user", "content": (
                f"Original English: {source}\nNepali Draft: {draft}\nBack-translation: {back_trans}\n\n"
                "Analysis: Compare original and back-translation.\nFixed Translation: Provide corrected Nepali."
            )}
        ]
        prompt = self.llama_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        outputs = self.llama_mod.generate(**inputs, max_new_tokens=400, do_sample=False)
        res = self.llama_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return res.split("Fixed Translation:")[-1].strip() if "Fixed Translation:" in res else res.strip()

# -----------------------------
# 3. Main Evaluation Execution
# -----------------------------
def run_advanced_eval(num_samples=50):
    proc = EchoRefineProcessor()
    auditor = SemanticAuditor()
    
    chrf = evaluate.load("chrf")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load("comet")

    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    eng_df = df[df['iso_639_3'] == 'eng'].reset_index()
    npi_df = df[df['iso_639_3'] == 'npi'].reset_index()

    eval_data = {"mBART": [], "Llama_Direct": [], "EchoRefine": [], "Ref": [], "Src": []}

    print(f"Evaluating {num_samples} samples...")
    for i in range(num_samples):
        src = eng_df.iloc[i]['text']
        ref = npi_df.iloc[i]['text']

        # Methods
        draft = proc.mbart_translate(src, "eng", "npi")
        direct = proc.llama_direct(src)
        
        current_refined = draft
        for _ in range(MAX_ITERATIONS):
            back = proc.mbart_translate(current_refined, "npi", "eng")
            if auditor.get_score(src, back) >= SEMANTIC_CONVERGENCE: break
            current_refined = proc.llama_reason_and_refine(src, current_refined, back)

        eval_data["mBART"].append(draft)
        eval_data["Llama_Direct"].append(direct)
        eval_data["EchoRefine"].append(current_refined)
        eval_data["Ref"].append(ref)
        eval_data["Src"].append(src)
        if (i+1) % 5 == 0: print(f"Progress: {i+1}/{num_samples}")

    # -----------------------------
    # 4. Scoring & Visualization
    # -----------------------------
    final_metrics = {}
    metric_names = ["chrF", "BLEU", "BERTScore", "COMET"]
    
    for model in ["mBART", "Llama_Direct", "EchoRefine"]:
        preds = eval_data[model]
        refs_nested = [[r] for r in eval_data["Ref"]]
        refs_flat = eval_data["Ref"]
        srcs = eval_data["Src"]

        ch = chrf.compute(predictions=preds, references=refs_nested)['score']
        bl = bleu.compute(predictions=preds, references=refs_nested)['bleu'] * 100
        bs = np.mean(bertscore.compute(predictions=preds, references=refs_flat, lang="ne")['f1']) * 100
        cm = comet.compute(predictions=preds, references=refs_flat, sources=srcs)['mean_score'] * 100
        
        final_metrics[model] = [ch, bl, bs, cm]

    # Plotting Logic
    x = np.arange(len(metric_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - width, final_metrics["mBART"], width, label='mBART (Baseline)', color='#1f77b4')
    ax.bar(x, final_metrics["Llama_Direct"], width, label='Llama-3.1-70B (Direct)', color='#ff7f0e')
    ax.bar(x + width, final_metrics["EchoRefine"], width, label='EchoRefine (Advanced BTI)', color='#2ca02c')

    ax.set_ylabel('Scores (Scaled 0-100)')
    ax.set_title(f'Translation Performance: English to Nepali (N={num_samples})')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Adding Value Labels on Top
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.savefig('echorefine_results.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'echorefine_results.png'")

if __name__ == "__main__":
    run_advanced_eval(num_samples=5)
