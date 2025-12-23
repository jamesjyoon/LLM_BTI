import os
import torch
import warnings
import numpy as np
import difflib
import matplotlib.pyplot as plt
import re
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Configuration
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
QE_MODEL_ID = "Unbabel/wmt22-comet-qe-da" # Reference-less Quality Estimation

# -----------------------------
# 1. Internal Judges (Reference-less)
# -----------------------------
class InternalJudge:
    def __init__(self):
        print("Loading COMET-QE Judge (Reference-less)...")
        # Load the QE model specifically for judging
        self.qe_metric = evaluate.load("comet", QE_MODEL_ID)

    def calculate_lcs(self, text1, text2):
        t1, t2 = text1.split(), text2.split()
        if not t1 or not t2: return 0.0
        return difflib.SequenceMatcher(None, t1, t2).ratio()

    def select_best(self, source, candidate_a, candidate_b):
        """
        Compares two candidates using QE.
        candidate_a: mBART Draft
        candidate_b: Llama Refined
        """
        # QE scores: higher is better
        results = self.qe_metric.compute(
            sources=[source, source],
            predictions=[candidate_a, candidate_b]
        )
        scores = results['scores']
        
        # Selection Logic: Accept refinement only if QE improves significantly
        # or if candidate_b is at least equal in quality.
        if scores[1] > scores[0]:
            return candidate_b, "LLM_Refined"
        return candidate_a, "mBART_Original"

# -----------------------------
# 2. Main System Class
# -----------------------------
class EchoRefineQE:
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

    def llama_refine(self, source, draft, back_trans):
        messages = [
            {"role": "system", "content": "You are a professional Nepali editor. Fix the draft based on the back-translation errors. Be concise. RESULT: [Nepali Only]"},
            {"role": "user", "content": f"Original: {source}\nDraft: {draft}\nBack-translation: {back_trans}\n\nRESULT:"}
        ]
        prompt = self.llama_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        outputs = self.llama_mod.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=self.llama_tok.eos_token_id)
        res = self.llama_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        final = res.split("RESULT:")[-1].strip()
        # Language Guard (Reject if English reasoning leaks)
        if len(re.findall(r'[a-zA-Z]', final)) > (len(final) * 0.2): return draft
        return final

# -----------------------------
# 3. Execution & Evaluation
# -----------------------------
def run_benchmark(num_samples=50):
    system = EchoRefineQE()
    judge = InternalJudge()
    
    # EXTERNAL METRICS (The ones we report)
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")
    bert = evaluate.load("bertscore")

    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    eng_df = df[df['iso_639_3'] == 'eng'].reset_index()
    npi_df = df[df['iso_639_3'] == 'npi'].reset_index()

    storage = {"mBART": [], "EchoRefine": [], "Ref": []}

    print(f"Starting benchmark on N={num_samples}...")

    for i in range(num_samples):
        src = eng_df.iloc[i]['text']
        ref = npi_df.iloc[i]['text']

        # 1. Draft
        draft = system.mbart_translate(src, "eng", "npi")
        
        # 2. Back-Translate
        back = system.mbart_translate(draft, "npi", "eng")
        
        # 3. Refine
        refined_candidate = system.llama_refine(src, draft, back)
        
        # 4. INTERNAL JUDGING (Does not see the 'ref' variable!)
        final_decision, choice_made = judge.select_best(src, draft, refined_candidate)

        storage["mBART"].append(draft)
        storage["EchoRefine"].append(final_decision)
        storage["Ref"].append(ref)
        
        if (i+1) % 5 == 0:
            print(f"[{i+1}/{num_samples}] Decision: {choice_made}")

    # -----------------------------
    # 4. Final Scoring
    # -----------------------------
    final_results = {}
    for key in ["mBART", "EchoRefine"]:
        preds = storage[key]
        refs_nested = [[r] for r in storage["Ref"]]
        
        b = bleu.compute(predictions=preds, references=refs_nested)['bleu'] * 100
        c = chrf.compute(predictions=preds, references=refs_nested)['score']
        bs = np.mean(bert.compute(predictions=preds, references=storage["Ref"], lang="ne")['f1']) * 100
        
        final_results[key] = [b, c, bs]

    # Print table
    print("\n" + "="*45)
    print(f"{'Metric':<15} | {'mBART':<10} | {'EchoRefine':<10}")
    print("-" * 45)
    metrics_list = ["BLEU", "chrF", "BERTScore"]
    for idx, m in enumerate(metrics_list):
        print(f"{m:<15} | {final_results['mBART'][idx]:<10.2f} | {final_results['EchoRefine'][idx]:<10.2f}")
    print("="*45)

    # Plot
    x = np.arange(len(metrics_list))
    width = 0.35
    plt.bar(x - width/2, final_results["mBART"], width, label='mBART Baseline')
    plt.bar(x + width/2, final_results["EchoRefine"], width, label='EchoRefine (QE-Judge)')
    plt.xticks(x, metrics_list); plt.ylabel("Score"); plt.legend(); plt.title("Final Performance Comparison")
    plt.savefig('qe_judged_results.png')

if __name__ == "__main__":
    run_benchmark(num_samples=5) # Increased to 100 for better significance
