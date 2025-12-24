import os
import torch
import warnings
import numpy as np
import difflib
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from comet import download_model, load_from_checkpoint

# --- Cluster Setup ---
import matplotlib
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Configuration
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct" 
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
QE_MODEL_NAME = "Unbabel/wmt22-comet-qe-da" # Reference-less QE

# -----------------------------
# 1. Internal Judge (Reference-less)
# -----------------------------
class InternalJudge:
    def __init__(self, token):
        print(f"Loading {QE_MODEL_NAME} Judge...")
        
        # Programmatically set the HF Token environment variable 
        # so comet (which uses huggingface_hub) can see it.
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HF_HUB_TOKEN"] = token

        try:
            # download_model does not take 'token' as an argument in recent versions
            model_path = download_model(QE_MODEL_NAME)
            self.qe_model = load_from_checkpoint(model_path)
            
            if torch.cuda.is_available():
                self.qe_model = self.qe_model.to("cuda")
        except Exception as e:
            print(f"Failed to load QE model: {e}")
            print("Note: Ensure you have accepted the license for wmt22-comet-qe-da on HF.")
            raise e

    def select_best(self, source, candidate_a, candidate_b):
        """Compares mBART Draft vs Llama Refined using QE."""
        data = [
            {"src": source, "mt": candidate_a},
            {"src": source, "mt": candidate_b}
        ]
        
        with torch.no_grad():
            # gpus=1 handles the distribution internally via comet
            outputs = self.qe_model.predict(data, batch_size=2, gpus=1 if torch.cuda.is_available() else 0)
        
        scores = outputs.scores
        
        if scores[1] > scores[0]:
            return candidate_b, "LLM_Refined", scores[1]
        return candidate_a, "mBART_Original", scores[0]

# -----------------------------
# 2. Translation System Class
# -----------------------------
class TranslationSystem:
    def __init__(self, token):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading {LLAMA_MODEL_ID}...")
        self.llama_tok = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, token=token)
        self.llama_mod = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_ID, 
            quantization_config=bnb_config, 
            device_map="auto", 
            token=token,
            dtype=torch.float16 # Fixed: replaced torch_dtype with dtype
        )
        
        print(f"Loading mBART-50...")
        self.nmt_tok = AutoTokenizer.from_pretrained(MBART_MODEL_ID)
        self.nmt_mod = AutoModelForSeq2SeqLM.from_pretrained(
            MBART_MODEL_ID, 
            dtype=torch.float16, # Fixed: replaced torch_dtype with dtype
            device_map="auto"
        )

    def mbart_translate(self, text, src_iso, tgt_iso):
        mapping = {"eng": "en_XX", "npi": "ne_NP"}
        self.nmt_tok.src_lang = mapping[src_iso]
        inputs = self.nmt_tok(text, return_tensors="pt", padding=True).to(self.nmt_mod.device)
        outputs = self.nmt_mod.generate(**inputs, forced_bos_token_id=self.nmt_tok.lang_code_to_id[mapping[tgt_iso]])
        return self.nmt_tok.decode(outputs[0], skip_special_tokens=True)

    def llama_refine(self, source, draft, back_trans):
        messages = [
            {"role": "system", "content": "You are a Nepali editor. Fix errors based on back-translation. Output ONLY Nepali text after 'RESULT:'"},
            {"role": "user", "content": f"Source: {source}\nDraft: {draft}\nBack-trans: {back_trans}\n\nRESULT:"}
        ]
        prompt = self.llama_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        outputs = self.llama_mod.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=self.llama_tok.eos_token_id)
        res = self.llama_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        final = res.split("RESULT:")[-1].strip()
        # Language Guard
        if len(re.findall(r'[a-zA-Z]', final)) > (len(final) * 0.25): return draft
        return final

# -----------------------------
# 3. Execution & Evaluation
# -----------------------------
def run_benchmark(num_samples=50):
    if not HUGGING_FACE_HUB_TOKEN:
        raise ValueError("HF Token not found. Check .env file.")

    system = TranslationSystem(HUGGING_FACE_HUB_TOKEN)
    judge = InternalJudge(HUGGING_FACE_HUB_TOKEN)
    
    # EXTERNAL METRICS
    metrics = {
        "bleu": evaluate.load("bleu"),
        "chrf": evaluate.load("chrf"),
        "bertscore": evaluate.load("bertscore")
    }

    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    eng_df = df[df['iso_639_3'] == 'eng'].reset_index()
    npi_df = df[df['iso_639_3'] == 'npi'].reset_index()

    storage = {"mBART": [], "EchoRefine": [], "Ref": []}

    print(f"Benchmarking {num_samples} samples...")

    for i in range(num_samples):
        src = eng_df.iloc[i]['text']
        ref = npi_df.iloc[i]['text']

        draft = system.mbart_translate(src, "eng", "npi")
        back = system.mbart_translate(draft, "npi", "eng")
        refined_candidate = system.llama_refine(src, draft, back)
        
        # Use QE judge to choose between original and refined
        final_decision, choice_made, qe_score = judge.select_best(src, draft, refined_candidate)

        storage["mBART"].append(draft)
        storage["EchoRefine"].append(final_decision)
        storage["Ref"].append(ref)
        
        if (i+1) % 5 == 0:
            print(f"[{i+1}/{num_samples}] Decision: {choice_made} (QE Score: {qe_score:.4f})")

    # Evaluation
    final_results = {}
    for key in ["mBART", "EchoRefine"]:
        preds = storage[key]
        refs_nested = [[r] for r in storage["Ref"]]
        
        b = metrics["bleu"].compute(predictions=preds, references=refs_nested)['bleu'] * 100
        c = metrics["chrf"].compute(predictions=preds, references=refs_nested)['score']
        bs = np.mean(metrics["bertscore"].compute(predictions=preds, references=storage["Ref"], lang="ne")['f1']) * 100
        final_results[key] = [b, c, bs]

    # Print Results
    print("\n" + "="*45)
    print(f"{'Metric':<15} | {'mBART':<10} | {'EchoRefine':<10}")
    print("-" * 45)
    labels = ["BLEU", "chrF", "BERTScore"]
    for idx, label in enumerate(labels):
        print(f"{label:<15} | {final_results['mBART'][idx]:<10.2f} | {final_results['EchoRefine'][idx]:<10.2f}")
    print("="*45)

    # Plot
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, final_results["mBART"], width, label='mBART Baseline')
    plt.bar(x + width/2, final_results["EchoRefine"], width, label='EchoRefine (QE-Judge)')
    plt.xticks(x, labels); plt.ylabel("Score"); plt.legend(); plt.title("Internal QE-Judging Evaluation")
    plt.savefig('qe_judged_benchmark.png')

if __name__ == "__main__":
    run_benchmark(num_samples=5)
