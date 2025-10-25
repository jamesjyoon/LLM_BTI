import os
os.environ["PYTHONUTF8"] = "1"
import logging
import warnings
import transformers

# Optionally, disable progress bars from the Hugging Face Hub.
os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"

# Set logging levels to ERROR so that only error messages (or above) are printed.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("evaluate").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Set Transformers logging to error.
transformers.logging.set_verbosity_error()

# Optionally, suppress all warnings.
warnings.filterwarnings("ignore")

from datasets import load_dataset
import evaluate
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import asyncio
from dotenv import load_dotenv
from portkey_ai import Portkey
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib

# -----------------------------
# Load API keys from .env file
# -----------------------------
load_dotenv()
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
GROQ_VIRTUAL_KEY = os.getenv("GROQ_VIRTUAL_KEY")
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# ----------------------------------------------------------------------------
# Asynchronous generate helper using Portkey's chat completions API
# ----------------------------------------------------------------------------
async def async_generate(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    client = Portkey(api_key=PORTKEY_API_KEY, virtual_key=GROQ_VIRTUAL_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    await asyncio.sleep(3)
    return response.choices[0].message.content.strip()

# -----------------------------
# NLLB Translator using facebook/nllb-200-distilled-600M
# -----------------------------
def get_nllb_code(language: str) -> str:
    """
    Map common language names to their NLLB language codes.
    For example:
      - "korean" -> "kor_Hang"
      - "english" -> "eng_Latn"
    """
    mapping = {
        "japanese": "jpn_Jpan",
        "irish": "gle_Latn",
        "nepali": "npi_Deva",
        "swahili": "swh_Latn",
        "burmese": "mya_Mymr",
        "swedish": "swe_Latn",
        "thai": "tha_Thai",
        "urdu": "urd_Arab",
        "chinese": "zh_Hant",
        "korean": "kor_Hang",
        "english": "eng_Latn"
    }
    
    return mapping.get(language.lower(), language)

class NLLBTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", name="NLLBTranslator", max_length=256, get_code_func=None):
        """
        get_code_func: a callable that accepts a language name (e.g., "English") and returns its NLLB language code (e.g., "eng_Latn").
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HUGGING_FACE_HUB_TOKEN
        )
        # Monkey-patch the tokenizer if it doesn't have lang_code_to_id.
        if not hasattr(self.tokenizer, "lang_code_to_id"):
            # Create a minimal mapping for the languages you are using.
            self.tokenizer.lang_code_to_id = {
                "jpn_Jpan": self.tokenizer.convert_tokens_to_ids("jpn_Jpan"),
                "npi_Deva": self.tokenizer.convert_tokens_to_ids("npi_Deva"),
                "swh_Latn": self.tokenizer.convert_tokens_to_ids("swh_Latn"),
                "tgk_Cyrl": self.tokenizer.convert_tokens_to_ids("tgk_Cyrl"),
                "gle_Latn": self.tokenizer.convert_tokens_to_ids("gle_Latn"),
                "mya_Mymr": self.tokenizer.convert_tokens_to_ids("mya_Mymr"),
                "swe_Latn": self.tokenizer.convert_tokens_to_ids("swe_Latn"),
                "tha_Thai": self.tokenizer.convert_tokens_to_ids("tha_Thai"),
                "urd_Arab": self.tokenizer.convert_tokens_to_ids("urd_Arab"),
                "zh_Hans": self.tokenizer.convert_tokens_to_ids("zh_Hans"),
                "zh_Hant": self.tokenizer.convert_tokens_to_ids("zh_Hant"),
                "kor_Hang": self.tokenizer.convert_tokens_to_ids("kor_Hang"),
                "eng_Latn": self.tokenizer.convert_tokens_to_ids("eng_Latn"),
                # add more language mappings as needed.
            }
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, token=HUGGING_FACE_HUB_TOKEN,
            # use_fast=False
        )
        self.max_length = max_length
        self.name = name
        self.get_code_func = get_code_func if get_code_func is not None else get_nllb_code

    def get_forced_bos_token_id(self, tgt_code: str) -> int:
        # If the tokenizer has the attribute, use it.
        if hasattr(self.tokenizer, "lang_code_to_id"):
            token_id = self.tokenizer.lang_code_to_id.get(tgt_code)
        else:
            token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Language token {tgt_code} not found in vocabulary.")
        return token_id

    def translate(self, source_text: str, src_lang: str, tgt_lang: str, num_candidates: int = 1):
        try:
            # Retrieve language codes dynamically.
            if self.get_code_func:
                src_code = self.get_code_func(src_lang)
                tgt_code = self.get_code_func(tgt_lang)
            else:
                src_code = src_lang
                tgt_code = tgt_lang
            # Set the tokenizer language attributes (if needed)
            self.tokenizer.src_lang = src_code
            self.tokenizer.tgt_lang = tgt_code

            inputs = self.tokenizer(source_text, return_tensors="pt", padding=True,
                                    truncation=True, max_length=self.max_length)
            # Compute forced_bos_token_id using the target language code.
            forced_bos_token_id = self.get_forced_bos_token_id(tgt_code)
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_return_sequences=num_candidates,
                do_sample=True
            )
            translations = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
            return translations[0] if num_candidates == 1 else translations
        except Exception as e:
            print(f"Error during NLLB translation: {e}")
            return "Translation Error (NLLB)"

# -----------------------------
# Llama3 Translator using Llama3 model
# -----------------------------
class Llama3Translator:
    def __init__(self, model="llama3-70b-8192", temperature=0.7, max_tokens=456):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def translate(self, source_text: str, src_lang: str, tgt_lang: str, num_candidates: int = 1):
        prompt = (
            f"Translate the following {src_lang} text to {tgt_lang}. "
            f"Provide only the translation without any additional commentary or explanation.\n\n"
            f"{src_lang} text: {source_text}\n\n"
            f"{tgt_lang} translation:"
        )
        try:
            translation = asyncio.run(async_generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ))
            # Post-process: extract text after the marker, if present
            if "translation:" in translation.lower():
                translation = translation.split("translation:", 1)[-1].strip()
            return translation
        except Exception as e:
            print(f"Error during Llama3 translation: {e}")
            return "Translation Error (Llama3)"

# -----------------------------
# Text Similarity Calculator
# -----------------------------
class TextSimilarityCalculator:
    @staticmethod
    def calculate_similarity(text1, text2):
        return SequenceMatcher(None, text1, text2).ratio()
    
    @staticmethod
    def extract_discrepancies(original, back_translation):
        original_sentences = [s.strip() for s in original.split('.') if s.strip()]
        back_sentences = [s.strip() for s in back_translation.split('.') if s.strip()]
        discrepancies = []
        max_sentences = min(len(original_sentences), len(back_sentences))
        for i in range(max_sentences):
            orig_sent = original_sentences[i]
            back_sent = back_sentences[i]
            similarity = TextSimilarityCalculator.calculate_similarity(orig_sent, back_sent)
            if similarity < 0.85:
                discrepancies.append({
                    "original": orig_sent,
                    "back_translation": back_sent,
                    "similarity": similarity
                })
        return discrepancies

# -----------------------------
# Enhanced Critique Agent using Llama3 (for refinement)
# -----------------------------
class EnhancedCritiqueAgent:
    def __init__(self, model="llama3-70b-8192", temperature=0.7, max_tokens=456):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        return asyncio.run(async_generate(prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens))

    def refine_with_discrepancies(self, translation: str, source_text: str, back_translation: str, discrepancies: list, src_lang: str, tgt_lang: str) -> str:
        discrepancy_text = ""
        for i, disc in enumerate(discrepancies):
            discrepancy_text += f"Discrepancy {i+1}:\n- Original: {disc['original']}\n- Back Translation: {disc['back_translation']}\n- Similarity: {disc['similarity']:.2f}\n\n"
        prompt = (
            f"Refine the following translation to address the discrepancies. Provide only the improved \
            {tgt_lang} translation without additional commentary.\n\n"
            f"Original {src_lang} Text:\n{source_text}\n\n"
            f"Current {tgt_lang} Translation:\n{translation}\n\n"
            f"Back Translation to {src_lang}:\n{back_translation}\n\n"
            f"Discrepancies:\n{discrepancy_text}\n"
            f"Improved {tgt_lang} Translation:"
        )
        try:
            improved_translation = self.generate(prompt)
            # Post-process: extract text after the marker, if present.
            if "Improved" in improved_translation:
                improved_translation = improved_translation.split("Improved", 1)[-1].strip()
            if "Explanation:" in improved_translation:
                improved_translation = improved_translation.split("Explanation:")[0].strip()
            return improved_translation
        except Exception as e:
            print(f"Error during refinement: {e}")
            return translation

# -----------------------------
# Convergent Back-Translation Refinement Agent
# -----------------------------
class ConvergentBackTranslationImprover:
    def __init__(self, translator, critique_agent):
        self.translator = translator  # Instance of NLLBTranslator
        self.critique_agent = critique_agent  # Llama3-based agent for refinement
        self.similarity_calculator = TextSimilarityCalculator()

    def improve_translation(self, source_text: str, src_lang: str, tgt_lang: str, convergence_threshold: float = 0.90,
                            max_iterations: int = 5, min_improvement_delta: float = 0.01, debug: bool = True) -> dict:
        results = {
            "iterations": [],
            "original_text": source_text,
            "initial_translation": None,
            "final_translation": None,
            "convergence_achieved": False,
            "similarity_scores": [],
            "translation_history": [],
            "back_translations": []
        }
        if debug:
            print("===== ORIGINAL SOURCE TEXT =====")
            print(source_text)
            print("==================================\n")
        primary_translation = self.translator.translate(source_text, src_lang, tgt_lang)
        results["initial_translation"] = primary_translation
        results["translation_history"].append(primary_translation)
        if debug:
            print("===== PRIMARY TRANSLATION =====")
            print(primary_translation)
            print("=================================\n")
        back_translation = self.translator.translate(primary_translation, tgt_lang, src_lang)
        results["back_translations"].append(back_translation)
        if debug:
            print("===== INITIAL BACK TRANSLATION =====")
            print(back_translation if back_translation else "[Empty back translation]")
            print("=====================================\n")
        initial_similarity = self.similarity_calculator.calculate_similarity(source_text, back_translation)
        results["similarity_scores"].append(initial_similarity)
        print(f"Initial similarity: {initial_similarity:.4f}")
        current_translation = primary_translation
        current_similarity = initial_similarity
        for iteration in range(1, max_iterations + 1):
            iteration_data = {
                "iteration": iteration,
                "translation": current_translation,
                "back_translation": back_translation,
                "similarity": current_similarity,
                "discrepancies": []
            }
            print(f"\n----- ITERATION {iteration} -----")
            discrepancies = self.similarity_calculator.extract_discrepancies(source_text, back_translation)
            iteration_data["discrepancies"] = discrepancies
            if debug:
                print(f"Found {len(discrepancies)} discrepancies:")
                for i, disc in enumerate(discrepancies):
                    print(f"Discrepancy {i+1}:")
                    print(f"Original: {disc['original']}")
                    print(f"Back Translation: {disc['back_translation']}")
                    print(f"Similarity: {disc['similarity']:.4f}\n")
            if current_similarity >= convergence_threshold:
                print(f"Convergence achieved with similarity {current_similarity:.4f} ≥ {convergence_threshold}")
                break
            refined_translation = self.critique_agent.refine_with_discrepancies(
                current_translation, source_text, back_translation, discrepancies, src_lang, tgt_lang
            )
            print("\nRefined Translation:")
            print(refined_translation)
            if refined_translation.strip() == current_translation.strip():
                print("No meaningful change detected. Ending iterations.\n")
                break
            previous_similarity = current_similarity
            current_translation = refined_translation
            results["translation_history"].append(current_translation)
            back_translation = self.translator.translate(current_translation, tgt_lang, src_lang)
            results["back_translations"].append(back_translation)
            if debug:
                print("\nUpdated Back Translation:")
                print(back_translation if back_translation else "[Empty back translation]")
            current_similarity = self.similarity_calculator.calculate_similarity(source_text, back_translation)
            results["similarity_scores"].append(current_similarity)
            print(f"\nNew similarity: {current_similarity:.4f}")
            iteration_data["similarity"] = current_similarity
            results["iterations"].append(iteration_data)
            if (current_similarity - previous_similarity) < min_improvement_delta:
                print(f"Improvement delta {(current_similarity - previous_similarity):.4f} is below threshold {min_improvement_delta}. Stopping iterations.\n")
                break
        results["final_translation"] = current_translation
        return results

# -----------------------------
# Translation Pipelines
# -----------------------------
def nllb_pipeline(source_text: str, source_lang: str, target_lang: str, translator_agent):
    return translator_agent.translate(source_text, src_lang=source_lang, tgt_lang=target_lang)

def convergent_back_translation_pipeline(source_text: str, source_lang: str, target_lang: str, improver_agent, max_iterations: int = 5):
    results = improver_agent.improve_translation(source_text, source_lang, target_lang, convergence_threshold=0.90, max_iterations=max_iterations, min_improvement_delta=0.01)
    return results["final_translation"]

# -----------------------------
# FloresTranslationEvaluator Class
# -----------------------------
class FloresTranslationEvaluator:
    def __init__(self, dataset_name="Muennighoff/flores200", lang_pair='eng_Latn-kor_Hang'):
        self.dataset_name = dataset_name
        self.lang_pair = lang_pair
        print(f"Loading dataset '{dataset_name}' with configuration '{lang_pair}' ...")
        # Load dataset using the configuration as a positional argument.
        self.dataset = load_dataset(self.dataset_name, self.lang_pair)
    
    def run_pipeline_on_example(self, index, translator, critique_agent, max_iterations=3, debug=True):
        # Determine which split to use: prefer 'test', then 'dev', then 'train'
        for candidate in ['test', 'dev', 'train']:
            if candidate in self.dataset:
                split = candidate
                break
        else:
            raise ValueError("No suitable split found in the dataset.")

        example = self.dataset[split][index]
        # Extract source and target language codes from the lang_pair string.
        src_code, tgt_code = self.lang_pair.split('-')
        
        # First, try to retrieve texts using a nested "translation" key if present.
        if "translation" in example:
            source_text = example["translation"].get(src_code)
            reference_translation = example["translation"].get(tgt_code)
        else:
            source_text = example.get(src_code)
            reference_translation = example.get(tgt_code)
        
        # If the texts are still not found, try with a "sentence_" prefix.
        if source_text is None:
            source_text = example.get("sentence_" + src_code)
        if reference_translation is None:
            reference_translation = example.get("sentence_" + tgt_code)
        
        if source_text is None or reference_translation is None:
            print("Available keys in example:", list(example.keys()))
            raise KeyError(f"Expected language keys '{src_code}' and/or '{tgt_code}' (or with 'sentence_' prefix) not found in dataset example.")

        print("Source text from dataset:")
        print(source_text)
        print("\nReference translation from dataset:")
        print(reference_translation)
        
        # Run convergent back-translation pipeline.
        convergent_improver = ConvergentBackTranslationImprover(translator, critique_agent)
        final_translation = convergent_back_translation_pipeline(source_text, "English", "Korean", convergent_improver, max_iterations)
        print("\nFinal Improved Translation from dataset example:")
        print(final_translation)
        
        # Get Llama3 translation
        llama3_translator = Llama3Translator()
        llama3_translation = llama3_translator.translate(source_text, "English", "Korean")
        print("\nLlama3 Translation:")
        print(llama3_translation)
        
        # Evaluation: compute COMET and chrF scores.
        comet = evaluate.load("comet", config_name="wmt20-comet-da")
        chrf = evaluate.load("chrf")
        
        # NLLB primary translation
        primary_translation = translator.translate(source_text, "English", "Korean")
        primary_comet = comet.compute(
            sources=[source_text],
            predictions=[primary_translation],
            references=[reference_translation]
        )
        primary_chrf = chrf.compute(
            predictions=[primary_translation],
            references=[reference_translation]
        )
        
        # Llama3 translation
        llama3_comet = comet.compute(
            sources=[source_text],
            predictions=[llama3_translation],
            references=[reference_translation]
        )
        llama3_chrf = chrf.compute(
            predictions=[llama3_translation],
            references=[reference_translation]
        )
        
        # Final improved translation
        final_comet = comet.compute(
            sources=[source_text],
            predictions=[final_translation],
            references=[reference_translation]
        )
        final_chrf = chrf.compute(
            predictions=[final_translation],
            references=[reference_translation]
        )
        
        primary_comet_score = primary_comet["scores"][0]
        llama3_comet_score = llama3_comet["scores"][0]
        final_comet_score = final_comet["scores"][0]
        
        print("\nEvaluation Scores for dataset example:")
        print("NLLB Primary Translation - COMET: {:.4f}, chrF: {:.4f}".format(primary_comet_score, primary_chrf["score"]))
        print("Llama3 Translation - COMET: {:.4f}, chrF: {:.4f}".format(llama3_comet_score, llama3_chrf["score"]))
        print("Final Improved Translation - COMET: {:.4f}, chrF: {:.4f}".format(final_comet_score, final_chrf["score"]))
        
        # Return the evaluation scores for aggregation.
        return {
            "primary_comet": primary_comet_score,
            "primary_chrf": primary_chrf["score"],
            "llama3_comet": llama3_comet_score,
            "llama3_chrf": llama3_chrf["score"],
            "final_comet": final_comet_score,
            "final_chrf": final_chrf["score"]
        }
    
    def run_pipeline_on_examples(self, src_lang:str, tgt_lang:str, indices, translator, critique_agent, max_iterations=3, debug=True):
        """Runs the pipeline on multiple examples, aggregates scores, prints average scores,
           and produces a single plot of the average COMET and chrF scores."""
        scores = {
            "primary_comet": [],
            "primary_chrf": [],
            "llama3_comet": [],
            "llama3_chrf": [],
            "final_comet": [],
            "final_chrf": []
        }
        for idx in indices:
            print("\n================== Running example {} ==================\n".format(idx))
            try:
                result = self.run_pipeline_on_example(idx, translator, critique_agent, max_iterations, debug)
                scores["primary_comet"].append(result["primary_comet"])
                scores["primary_chrf"].append(result["primary_chrf"])
                scores["llama3_comet"].append(result["llama3_comet"])
                scores["llama3_chrf"].append(result["llama3_chrf"])
                scores["final_comet"].append(result["final_comet"])
                scores["final_chrf"].append(result["final_chrf"])
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
        
        # Compute average scores.
        n = len(scores["final_comet"])
        if n == 0:
            print("No scores to average.")
            return
        
        avg_primary_comet = sum(scores["primary_comet"]) / n
        avg_primary_chrf = sum(scores["primary_chrf"]) / n
        avg_llama3_comet = sum(scores["llama3_comet"]) / n
        avg_llama3_chrf = sum(scores["llama3_chrf"]) / n
        avg_final_comet = sum(scores["final_comet"]) / n
        avg_final_chrf = sum(scores["final_chrf"]) / n

        print("\n================== Average Evaluation Scores over {} examples ==================".format(n))
        print("Average NLLB Primary Translation - COMET: {:.4f}, chrF: {:.4f}".format(avg_primary_comet, avg_primary_chrf))
        print("Average Llama3 Translation - COMET: {:.4f}, chrF: {:.4f}".format(avg_llama3_comet, avg_llama3_chrf))
        print("Average Final Improved Translation - COMET: {:.4f}, chrF: {:.4f}".format(avg_final_comet, avg_final_chrf))
        
        # Produce a single plot with the average scores.
        labels = ['COMET', 'chrF']
        primary_avg_scores = [avg_primary_comet, avg_primary_chrf]
        llama3_avg_scores = [avg_llama3_comet, avg_llama3_chrf]
        final_avg_scores = [avg_final_comet, avg_final_chrf]
        x = np.arange(len(labels))
        width = 0.2
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, primary_avg_scores, width, label='NLLB Primary Translation')
        rects2 = ax.bar(x, llama3_avg_scores, width, label='Llama3 Translation')
        rects3 = ax.bar(x + width, final_avg_scores, width, label='Final Improved Translation')
        
        ax.set_ylabel('Average Scores')
        ax.set_title(f'{src_lang} to {tgt_lang}'.format(n))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        fig.tight_layout()
        plt.show()

# -----------------------------
# Main Execution: Evaluate on 5 Dataset Examples
# -----------------------------
if __name__ == "__main__":
    # Initialize the NLLB translator with explicit language code mapping.
    nllb_translator = NLLBTranslator(get_code_func=get_nllb_code)
    # Initialize the critique agent (Llama3-based for refinement).
    critique_agent = EnhancedCritiqueAgent(model="llama3-70b-8192")
    
    # Create an instance of the FloresTranslationEvaluator using the default configuration.
    flores_evaluator = FloresTranslationEvaluator()  # Defaults to "Muennighoff/flores200" and "kor_Hang-eng_Latn"
    
    # Run the pipeline on the first 5 examples from the available split and produce a plot with average scores.
    indices_to_use = list(range(10))
    flores_evaluator.run_pipeline_on_examples("English", "Korean", indices_to_use, nllb_translator, critique_agent, max_iterations=3)
