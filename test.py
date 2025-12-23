import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

# --- NEW: Semantic Similarity Calculator ---
class SemanticTrigger:
    def __init__(self):
        # Using a tiny, fast model for the verification step
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def check(self, original, back_trans):
        emb1 = self.model.encode(original, convert_to_tensor=True)
        emb2 = self.model.encode(back_trans, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

# --- IMPROVED: Reasoning-Based Refiner ---
class AdvancedBTIProcessor:
    # ... (Keep previous __init__ and mbart_translate methods) ...

    def llama_advanced_refine(self, source, draft, back_trans):
        """Uses Chain-of-Thought to analyze errors before fixing them."""
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a linguistic auditor specializing in English and Nepali. "
            f"Your goal is to compare an 'Original English' sentence with a 'Back-translated English' sentence "
            f"to identify meaning shifts in the 'Nepali Draft'.\n\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Original English: {source}\n"
            f"Current Nepali Draft: {draft}\n"
            f"Back-translation into English: {back_trans}\n\n"
            f"Step 1: Briefly list any contradictions or missing info between the Original and the Back-translation.\n"
            f"Step 2: Provide the final, corrected Nepali translation.\n\n"
            f"Format your response as:\n"
            f"Analysis: [your analysis]\n"
            f"Fixed Translation: [Nepali text only]<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"Analysis:"
        )
        
        # Use greedy decoding for Step 1 & 2
        inputs = self.llama_tok(prompt, return_tensors="pt").to(self.llama_mod.device)
        outputs = self.llama_mod.generate(**inputs, max_new_tokens=300, do_sample=False)
        full_text = self.llama_tok.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the portion after "Fixed Translation:"
        if "Fixed Translation:" in full_text:
            return full_text.split("Fixed Translation:")[-1].strip()
        return full_text.strip()

# --- OPTIMIZED: The "Smart" Loop ---
def run_advanced_bti(system, semantic_checker, source_text):
    # 1. Start with the strongest base (mBART)
    current_nepali = system.mbart_translate(source_text, "eng", "npi")
    
    # 2. Iterative loop with Semantic Exit
    for _ in range(2): # 2 iterations is usually the 'Sweet Spot' for 70B models
        back_trans = system.mbart_translate(current_nepali, "npi", "eng")
        
        # Check meaning, not just word overlap
        semantic_score = semantic_checker.check(source_text, back_trans)
        
        if semantic_score > 0.95: # Very high semantic match
            break
            
        # Refine using Reasoning
        current_nepali = system.llama_advanced_refine(source_text, current_nepali, back_trans)
        
    return current_nepali
