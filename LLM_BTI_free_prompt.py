import os
import logging
import warnings
import asyncio
from difflib import SequenceMatcher

import evaluate
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import transformers
from datasets import load_dataset
from dotenv import load_dotenv
from portkey_ai import Portkey
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure UTF-8 and disable HF Hub progress bars.
os.environ["PYTHONUTF8"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"

# Set logging levels so only errors are emitted.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("evaluate").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

# Optionally suppress warnings.
warnings.filterwarnings("ignore")

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
        temperature=temperature,
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
        "french": "fra_Latn",
        "irish": "gle_Latn",
        "nepali": "npi_Deva",
        "swahili": "swh_Latn",
        "burmese": "mya_Mymr",
        "swedish": "swe_Latn",
        "thai": "tha_Thai",
        "urdu": "urd_Arab",
        "chinese": "zh_Hant",
        "korean": "kor_Hang",
        "english": "eng_Latn",
    }
    return mapping.get(language.lower(), language)


class NLLBTranslator:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        name: str = "NLLBTranslator",
        max_length: int = 256,
        get_code_func=None,
    ):
        """
        get_code_func: a callable that accepts a language name (e.g., "English")
        and returns its NLLB language code (e.g., "eng_Latn").
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HUGGING_FACE_HUB_TOKEN
        )
        # Monkey-patch the tokenizer if it doesn't have lang_code_to_id.
        if not hasattr(self.tokenizer, "lang_code_to_id"):
            self.tokenizer.lang_code_to_id = {
                "fra_Latn": self.tokenizer.convert_tokens_to_ids("fra_Latn"),
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
            }
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, token=HUGGING_FACE_HUB_TOKEN
        )
        self.max_length = max_length
        self.name = name
        self.get_code_func = get_code_func if get_code_func is not None else get_nllb_code

    def get_forced_bos_token_id(self, tgt_code: str) -> int:
        if hasattr(self.tokenizer, "lang_code_to_id"):
            token_id = self.tokenizer.lang_code_to_id.get(tgt_code)
        else:
            token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Language token {tgt_code} not found in vocabulary.")
        return token_id

    def translate(
        self,
        source_text: str,
        src_lang: str,
        tgt_lang: str,
        num_candidates: int = 1,
    ):
        try:
            src_code = self.get_code_func(src_lang)
            tgt_code = self.get_code_func(tgt_lang)
            self.tokenizer.src_lang = src_code
            self.tokenizer.tgt_lang = tgt_code

            inputs = self.tokenizer(
                source_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            forced_bos_token_id = self.get_forced_bos_token_id(tgt_code)
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_return_sequences=num_candidates,
                do_sample=True,
            )
            translations = [
                self.tokenizer.decode(o, skip_special_tokens=True).strip()
                for o in outputs
            ]
            return translations[0] if num_candidates == 1 else translations
        except Exception as e:  # pragma: no cover - defensive
            print(f"Error during NLLB translation: {e}")
            return "Translation Error (NLLB)"


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
            similarity = TextSimilarityCalculator.calculate_similarity(
                orig_sent, back_sent
            )
            if similarity < 0.85:
                discrepancies.append(
                    {
                        "original": orig_sent,
                        "back_translation": back_sent,
                        "similarity": similarity,
                    }
                )
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
        return asyncio.run(
            async_generate(
                prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

    def refine_with_discrepancies(
        self,
        translation: str,
        source_text: str,
        back_translation: str,
        discrepancies: list,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        discrepancy_text = ""
        for i, disc in enumerate(discrepancies):
            discrepancy_text += (
                f"Discrepancy {i+1}:\n- Original: {disc['original']}\n"
                f"- Back Translation: {disc['back_translation']}\n"
                f"- Similarity: {disc['similarity']:.2f}\n\n"
            )
        prompt = (
            f"Refine the following translation to address the discrepancies. "
            f"Provide only the improved {tgt_lang} translation without additional "
            f"commentary.\n\n"
            f"Original {src_lang} Text:\n{source_text}\n\n"
            f"Current {tgt_lang} Translation:\n{translation}\n\n"
            f"Back Translation to {src_lang}:\n{back_translation}\n\n"
            f"Discrepancies:\n{discrepancy_text}\n"
            f"Improved {tgt_lang} Translation:"
        )
        try:
            improved_translation = self.generate(prompt)
            if "Improved" in improved_translation:
                improved_translation = improved_translation.split("Improved", 1)[-1].strip()
            if "Explanation:" in improved_translation:
                improved_translation = improved_translation.split("Explanation:")[0].strip()
            return improved_translation
        except Exception as e:  # pragma: no cover - defensive
            print(f"Error during refinement: {e}")
            return translation


# -----------------------------
# Convergent Back-Translation Refinement Agent
# -----------------------------
class ConvergentBackTranslationImprover:
    def __init__(self, translator, critique_agent):
        self.translator = translator
        self.critique_agent = critique_agent
        self.similarity_calculator = TextSimilarityCalculator()

    def improve_translation(
        self,
        source_text: str,
        src_lang: str,
        tgt_lang: str,
        convergence_threshold: float = 0.90,
        max_iterations: int = 5,
        min_improvement_delta: float = 0.01,
        debug: bool = True,
    ) -> dict:
        results = {
            "iterations": [],
            "original_text": source_text,
            "initial_translation": None,
            "final_translation": None,
            "convergence_achieved": False,
            "similarity_scores": [],
            "translation_history": [],
            "back_translations": [],
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
        initial_similarity = self.similarity_calculator.calculate_similarity(
            source_text, back_translation
        )
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
                "discrepancies": [],
            }
            print(f"\n----- ITERATION {iteration} -----")
            discrepancies = self.similarity_calculator.extract_discrepancies(
                source_text, back_translation
            )
            iteration_data["discrepancies"] = discrepancies
            if debug:
                print(f"Found {len(discrepancies)} discrepancies:")
                for i, disc in enumerate(discrepancies):
                    print(f"Discrepancy {i+1}:")
                    print(f"Original: {disc['original']}")
                    print(f"Back Translation: {disc['back_translation']}")
                    print(f"Similarity: {disc['similarity']:.4f}\n")
            if current_similarity >= convergence_threshold:
                print(
                    "Convergence achieved with similarity "
                    f"{current_similarity:.4f} ≥ {convergence_threshold}"
                )
                break
            refined_translation = self.critique_agent.refine_with_discrepancies(
                current_translation,
                source_text,
                back_translation,
                discrepancies,
                src_lang,
                tgt_lang,
            )
            print("\nRefined Translation:")
            print(refined_translation)
            if refined_translation.strip() == current_translation.strip():
                print("No meaningful change detected. Ending iterations.\n")
                break
            previous_similarity = current_similarity
            current_translation = refined_translation
            results["translation_history"].append(current_translation)
            back_translation = self.translator.translate(
                current_translation, tgt_lang, src_lang
            )
            results["back_translations"].append(back_translation)
            if debug:
                print("\nUpdated Back Translation:")
                print(back_translation if back_translation else "[Empty back translation]")
            current_similarity = self.similarity_calculator.calculate_similarity(
                source_text, back_translation
            )
            results["similarity_scores"].append(current_similarity)
            print(f"\nNew similarity: {current_similarity:.4f}")
            iteration_data["similarity"] = current_similarity
            results["iterations"].append(iteration_data)
            if (current_similarity - previous_similarity) < min_improvement_delta:
                print(
                    "Improvement delta "
                    f"{(current_similarity - previous_similarity):.4f} is below threshold "
                    f"{min_improvement_delta}. Stopping iterations.\n"
                )
                break
        results["final_translation"] = current_translation
        return results


# -----------------------------
# Translation Pipelines
# -----------------------------
def nllb_pipeline(source_text: str, source_lang: str, target_lang: str, translator_agent):
    return translator_agent.translate(source_text, src_lang=source_lang, tgt_lang=target_lang)


def convergent_back_translation_pipeline(
    source_text: str,
    source_lang: str,
    target_lang: str,
    improver_agent,
    max_iterations: int = 5,
):
    results = improver_agent.improve_translation(
        source_text,
        source_lang,
        target_lang,
        convergence_threshold=0.90,
        max_iterations=max_iterations,
        min_improvement_delta=0.01,
    )
    return results["final_translation"]


# -----------------------------
# Main Execution: User Input for Backtranslation Evaluation
# -----------------------------
if __name__ == "__main__":
    # Initialize the NLLB translator with explicit language code mapping.
    nllb_translator = NLLBTranslator(get_code_func=get_nllb_code)
    # Initialize the critique agent (Llama3-based for refinement).
    critique_agent = EnhancedCritiqueAgent(model="llama3-70b-8192")

    # Create a convergent back-translation improver.
    convergent_improver = ConvergentBackTranslationImprover(
        nllb_translator, critique_agent
    )

    # Get user input for the sentence and language choices.
    print("Enter the source text for backtranslation evaluation:")
    user_source_text = input("> ")
    src_lang = input("Enter source language (default English): ") or "English"
    tgt_lang = input("Enter target language (default Chinese): ") or "Chinese"

    print("\n===== ORIGINAL SOURCE TEXT =====")
    print(user_source_text)
    print("================================\n")
    print("Performing translation and iterative backtranslation...\n")

    # Run the convergent back translation pipeline on the user-provided sentence.
    final_translation = convergent_back_translation_pipeline(
        user_source_text, src_lang, tgt_lang, convergent_improver, max_iterations=3
    )

    print("\n===== FINAL IMPROVED TRANSLATION =====")
    print(final_translation)
    print("======================================\n")