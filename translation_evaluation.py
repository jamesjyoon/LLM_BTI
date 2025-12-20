import os
import logging
import warnings
import transformers
import time
import asyncio
from dotenv import load_dotenv
from portkey_ai import Portkey
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Import for sentence-transformers
from sentence_transformers import SentenceTransformer, util

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set backend for matplotlib
matplotlib.use('Agg') # Use 'Agg' for non-interactive environments, 'TkAgg' for interactive plotting

# Optionally, disable progress bars from the Hugging Face Hub.
os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"

# Set logging levels to ERROR so that only error messages (or above) are printed.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("evaluate").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR) # Suppress SBERT warnings

# Set Transformers logging to error.
transformers.logging.set_verbosity_error()

# Optionally, suppress all warnings.
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
async def async_generate(prompt: str, model: str, temperature: float, max_tokens: int) -> tuple:
    start_time = time.perf_counter()
    client = Portkey(api_key=PORTKEY_API_KEY, virtual_key=GROQ_VIRTUAL_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    # Adding a small delay to avoid hitting rate limits if calls are too frequent
    await asyncio.sleep(0.5) 
    end_time = time.perf_counter()
    duration = end_time - start_time
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    return response.choices[0].message.content.strip(), duration, prompt_tokens, completion_tokens

# -----------------------------
# Language Code Mappers (NLLB and mBART)
# -----------------------------
def get_nllb_code(language: str) -> str:
    """
    Map common language names to their NLLB language codes.
    NLLB (No Language Left Behind)
    """
    mapping = {
        "japanese": "jpn_Jpan", "irish": "gle_Latn", "nepali": "npi_Deva",
        "swahili": "swh_Latn", "burmese": "mya_Mymr", "swedish": "swe_Latn",
        "thai": "tha_Thai", "urdu": "urd_Arab", "arabic": "arb_Arab",
        "chinese (simplified)": "zho_Hans", "chinese (traditional)": "zho_Hant",
        "korean": "kor_Hang", "english": "eng_Latn", "french": "fra_Latn",
        "spanish": "spa_Latn", "russian": "rus_Cyrl", "hindi": "hin_Deva",
        "bengali": "ben_Beng", "amharic": "amh_Ethi", "vietnamese": "vie_Latn",
        "tagalog": "tgl_Latn", "gujarati": "guj_Gujr", "lao": "lao_Laoo",
        "oriya": "ory_Orya", "assamese": "asm_Beng", "khmer": "khm_Khmr",
        "malayalam": "mal_Mlym", "marathi": "mar_Deva", "maithili": "mai_Deva",
        "telugu": "tel_Telu", "tamil": "tam_Taml", "sanskrit": "san_Deva",
        "tibetan": "bod_Tibt", "georgian": "kat_Geor", "kazakh": "kaz_Cyrl",
        "kyrgyz": "kir_Cyrl", "mongolian": "mon_Cyrl", "uzbek": "uzb_Latn",
        "pashto": "pbu_Arab", "persian": "pes_Arab", "punjabi": "pan_Guru",
        "sindhi": "snd_Arab", "azerbaijani": "azj_Latn", "bashkir": "bak_Cyrl",
        "belarusian": "bel_Cyrl", "bulgarian": "bul_Cyrl", "catalan": "cat_Latn",
        "cebuano": "ceb_Latn", "croatian": "hrv_Latn", "czech": "ces_Latn",
        "danish": "dan_Latn", "dutch": "nld_Latn", "estonian": "est_Latn",
        "finnish": "fin_Latn", "german": "deu_Latn", "greek": "ell_Grek",
        "hebrew": "heb_Hebr", "hungarian": "hun_Latn", "icelandic": "isl_Latn",
        "indonesian": "ind_Latn", "italian": "ita_Latn", "latvian": "lav_Latn",
        "lithuanian": "lit_Latn", "macedonian": "mkd_Cyrl", "malagasy": "plt_Latn",
        "malay": "zsm_Latn", "maltese": "mlt_Latn", "norwegian": "nno_Latn",
        "polish": "pol_Latn", "portuguese": "por_Latn", "romanian": "ron_Latn",
        "serbian": "srp_Cyrl", "slovak": "slk_Latn", "slovenian": "slv_Latn",
        "thai": "tha_Thai", "turkish": "tur_Latn", "ukrainian": "ukr_Cyrl",
        "vietnamese": "vie_Latn", "welsh": "cym_Latn",
    }
    # Handle direct codes if passed
    if language.lower() in [v.lower() for v in mapping.values()]: return language
    return mapping.get(language.lower(), language)

def get_nllb_lang_name(nllb_code: str) -> str:
    """
    Map NLLB language codes back to common language names for display.
    """
    reverse_mapping = {
        "jpn_Jpan": "Japanese", "gle_Latn": "Irish", "npi_Deva": "Nepali",
        "swh_Latn": "Swahili", "mya_Mymr": "Burmese", "swe_Latn": "Swedish",
        "tha_Thai": "Thai", "urd_Arab": "Urdu", "arb_Arab": "Arabic",
        "zho_Hans": "Chinese (Simplified)", "zho_Hant": "Chinese (Traditional)",
        "kor_Hang": "Korean", "eng_Latn": "English", "fra_Latn": "French",
        "spa_Latn": "Spanish", "rus_Cyrl": "Russian", "hin_Deva": "Hindi",
        "ben_Beng": "Bengali", "amh_Ethi": "Amharic", "vie_Latn": "Vietnamese",
        "tgk_Cyrl": "Tajik", "tgl_Latn": "Tagalog", "guj_Gujr": "Gujarati",
        "lao_Laoo": "Lao", "ory_Orya": "Oriya", "asm_Beng": "Assamese",
        "khm_Khmr": "Khmer", "mal_Mlym": "Malayalam", "mar_Deva": "Marathi",
        "mai_Deva": "Maithili", "tel_Telu": "Telugu", "tam_Taml": "Tamil",
        "san_Deva": "Sanskrit", "bod_Tibt": "Tibetan", "kat_Geor": "Georgian",
        "kaz_Cyrl": "Kazakh", "kir_Cyrl": "Kyrgyz", "mon_Cyrl": "Mongolian",
        "uzb_Latn": "Uzbek", "pbu_Arab": "Pashto", "pes_Arab": "Persian",
        "pan_Guru": "Punjabi", "snd_Arab": "Sindhi", "azj_Latn": "Azerbaijani",
        "bak_Cyrl": "Bashkir", "bel_Cyrl": "Belarusian", "bul_Cyrl": "Bulgarian",
        "cat_Latn": "Catalan", "ceb_Latn": "Cebuano", "hrv_Latn": "Croatian",
        "ces_Latn": "Czech", "dan_Latn": "Danish", "nld_Latn": "Dutch",
        "est_Latn": "Estonian", "fin_Latn": "Finnish", "deu_Latn": "German",
        "ell_Grek": "Greek", "heb_Hebr": "Hebrew", "hun_Latn": "Hungarian",
        "isl_Latn": "Icelandic", "ind_Latn": "Indonesian", "ita_Latn": "Italian",
        "lav_Latn": "Latvian", "lit_Latn": "Lithuanian", "mkd_Cyrl": "Macedonian",
        "plt_Latn": "Malagasy", "zsm_Latn": "Malay", "mlt_Latn": "Maltese",
        "nno_Latn": "Norwegian", "pol_Latn": "Polish", "por_Latn": "Portuguese",
        "ron_Latn": "Romanian", "srp_Cyrl": "Serbian", "slk_Latn": "Slovak",
        "slv_Latn": "Slovenian", "tur_Latn": "Turkish", "ukr_Cyrl": "Ukrainian",
        "cym_Latn": "Welsh",
    }
    return reverse_mapping.get(nllb_code, nllb_code)

def get_mbart_code(language: str) -> str:
    """
    Map common language names to their mBART language codes.
    mBART codes are slightly different, e.g., 'en_XX' for English.
    """
    mapping = {
        "japanese": "ja_XX", "irish": "ga_IE", "nepali": "ne_NP",
        "swahili": "sw_KE", "burmese": "my_MM", "swedish": "sv_SE",
        "thai": "th_TH", "urdu": "ur_PK", "arabic": "ar_AR",
        "chinese (simplified)": "zh_CN", "chinese (traditional)": "zh_TW",
        "korean": "ko_KR", "english": "en_XX", "french": "fr_XX",
        "spanish": "es_XX", "russian": "ru_RU", "hindi": "hi_IN",
        "bengali": "bn_IN", "amharic": "am_ET", "vietnamese": "vi_VN",
        "tagalog": "tl_XX", # mBART doesn't have Tajik, using Tagalog as general fallback
        # Add more mBART codes as needed
    }
    # Handle direct codes if passed
    if language.lower() in [v.lower() for v in mapping.values()]: return language
    return mapping.get(language.lower(), language)

def get_mbart_lang_name(mbart_code: str) -> str:
    """
    Map mBART language codes back to common language names for display.
    """
    reverse_mapping = {
        "ja_XX": "Japanese", "ga_IE": "Irish", "ne_NP": "Nepali",
        "sw_KE": "Swahili", "my_MM": "Burmese", "sv_SE": "Swedish",
        "th_TH": "Thai", "ur_PK": "Urdu", "ar_AR": "Arabic",
        "zh_CN": "Chinese (Simplified)", "zh_TW": "Chinese (Traditional)",
        "ko_KR": "Korean", "en_XX": "English", "fr_XX": "French",
        "es_XX": "Spanish", "ru_RU": "Russian", "hi_IN": "Hindi",
        "bn_IN": "Bengali", "am_ET": "Amharic", "vi_VN": "Vietnamese",
        "tl_XX": "Tagalog",
    }
    return reverse_mapping.get(mbart_code, mbart_code)

# -----------------------------
# Translator Classes
# -----------------------------
class BaseTranslator:
    def __init__(self, name="BaseTranslator", max_length=512):
        self.name = name
        self.max_length = max_length
        self.total_duration = 0
        self.total_tokens = 0

    def translate(self, source_text: str, src_lang: str, tgt_lang: str, num_candidates: int = 1):
        raise NotImplementedError("Subclasses must implement 'translate' method.")

class NLLBTranslator(BaseTranslator):
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", name="NLLBTranslator", max_length=512, get_code_func=None):
        super().__init__(name, max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_HUB_TOKEN)
        # # Populate lang_code_to_id if missing (for older tokenizer versions)
        # if not hasattr(self.tokenizer, "lang_code_to_id"):
        #      self.tokenizer.lang_code_to_id = {get_nllb_code(lang): self.tokenizer.convert_tokens_to_ids(get_nllb_code(lang)) for lang in get_nllb_code('').keys()}
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HUGGING_FACE_HUB_TOKEN)
        self.get_code_func = get_code_func if get_code_func is not None else get_nllb_code

    def get_forced_bos_token_id(self, tgt_code: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Language token {tgt_code} not found in NLLB vocabulary.")
        return token_id

    def translate(self, source_text: str, src_lang: str, tgt_lang: str, num_candidates: int = 1):
        start_time = time.perf_counter()
        try:
            src_code = self.get_code_func(src_lang)
            tgt_code = self.get_code_func(tgt_lang)
            
            self.tokenizer.src_lang = src_code # Set source language for tokenization
            inputs = self.tokenizer(source_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            
            forced_bos_token_id = self.get_forced_bos_token_id(tgt_code)
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_return_sequences=num_candidates,
                do_sample=False # For deterministic output in evaluation
            )
            translations = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.total_duration += duration
            self.total_tokens += inputs.input_ids.numel() + sum(len(o) for o in outputs)

            return translations[0] if num_candidates == 1 else translations
        except Exception as e:
            print(f"Error during NLLB translation: {e}")
            return "Translation Error (NLLB)"

class MBARTTranslator(BaseTranslator):
    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt", name="MBARTTranslator", max_length=512, get_code_func=None):
        super().__init__(name, max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_HUB_TOKEN)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HUGGING_FACE_HUB_TOKEN)
        self.get_code_func = get_code_func if get_code_func is not None else get_mbart_code

    def translate(self, source_text: str, src_lang: str, tgt_lang: str, num_candidates: int = 1):
        start_time = time.perf_counter()
        try:
            src_code = self.get_code_func(src_lang)
            tgt_code = self.get_code_func(tgt_lang)

            self.tokenizer.src_lang = src_code # Set source language for tokenization
            inputs = self.tokenizer(source_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            
            # mBART uses tgt_lang_code as forced_bos_token_id
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                max_length=self.max_length,
                num_return_sequences=num_candidates,
                do_sample=False
            )
            translations = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]

            end_time = time.perf_counter()
            duration = end_time - start_time
            self.total_duration += duration
            self.total_tokens += inputs.input_ids.numel() + sum(len(o) for o in outputs)

            return translations[0] if num_candidates == 1 else translations
        except Exception as e:
            print(f"Error during mBART translation: {e}")
            return "Translation Error (mBART)"


class LLMOnlyTranslator(BaseTranslator):
    def __init__(self, model="llama-3.3-70b-versatile", name="LLMOnlyTranslator", temperature=0.7, max_tokens=512):
        super().__init__(name, max_tokens) # max_tokens passed as max_length to BaseTranslator
        self.model = model
        self.temperature = temperature

    def translate(self, source_text: str, src_lang: str, tgt_lang: str, num_candidates: int = 1):
        # num_candidates is ignored for LLM-based translation for simplicity as it's not directly supported in async_generate
        prompt = (
            f"Translate the following {src_lang} text to {tgt_lang}. "
            f"Provide only the translation without any additional commentary or explanation.\n\n"
            f"{src_lang} text: {source_text}\n\n"
            f"{tgt_lang} translation:"
        )
        try:
            translation, duration, prompt_tokens, completion_tokens = asyncio.run(async_generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_length # Use max_length from BaseTranslator
            ))
            self.total_duration += duration
            self.total_tokens += prompt_tokens + completion_tokens

            # Post-process: extract text after the marker, if present
            if "translation:" in translation.lower():
                translation = translation.split("translation:", 1)[-1].strip()
            return translation
        except Exception as e:
            print(f"Error during LLM direct translation: {e}")
            return "Translation Error (LLM Direct)"


# -----------------------------
# Similarity Calculators
# -----------------------------
class SimilarityCalculator:
    def calculate_similarity(self, text1: str, text2: str) -> float:
        raise NotImplementedError("Subclasses must implement 'calculate_similarity' method.")

    def extract_discrepancies(self, original: str, back_translation: str) -> list:
        raise NotImplementedError("Subclasses must implement 'extract_discrepancies' method.")

class LCSBasedSimilarityCalculator(SimilarityCalculator):
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculates character-level similarity using Longest Common Subsequence (LCS) ratio."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def extract_discrepancies(self, original: str, back_translation: str, similarity_threshold: float = 0.85) -> list:
        original_sentences = [s.strip() for s in original.split('.') if s.strip()]
        back_sentences = [s.strip() for s in back_translation.split('.') if s.strip()]
        discrepancies = []
        # Attempt to match sentences to find discrepancies
        for orig_sent in original_sentences:
            best_match_back_sent = None
            max_similarity = 0
            for back_sent in back_sentences:
                similarity = self.calculate_similarity(orig_sent, back_sent)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_back_sent = back_sent
            
            if max_similarity < similarity_threshold and best_match_back_sent: # Threshold for considering a discrepancy
                discrepancies.append({
                    "original": orig_sent,
                    "back_translation": best_match_back_sent,
                    "similarity": max_similarity
                })
        return discrepancies

class SentenceTransformerSimilarityCalculator(SimilarityCalculator):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between sentence embeddings."""
        if not text1 or not text2: return 0.0 # Handle empty strings
        embeddings1 = self.model.encode(text1, convert_to_tensor=True)
        embeddings2 = self.model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings1, embeddings2).item()
    
    def extract_discrepancies(self, original: str, back_translation: str, similarity_threshold: float = 0.75) -> list:
        original_sentences = [s.strip() for s in original.split('.') if s.strip()]
        back_sentences = [s.strip() for s in back_translation.split('.') if s.strip()]
        discrepancies = []
        
        # Using sentence embeddings for matching sentences
        if not original_sentences or not back_sentences: return []

        orig_embeddings = self.model.encode(original_sentences, convert_to_tensor=True)
        back_embeddings = self.model.encode(back_sentences, convert_to_tensor=True)

        # Compute cosine similarities between all pairs of sentences
        cosine_scores = util.pytorch_cos_sim(orig_embeddings, back_embeddings)

        for i, orig_sent in enumerate(original_sentences):
            # Find the best matching back_translation sentence
            best_match_idx = cosine_scores[i].argmax().item()
            best_match_back_sent = back_sentences[best_match_idx]
            max_similarity = cosine_scores[i][best_match_idx].item()

            if max_similarity < similarity_threshold:
                discrepancies.append({
                    "original": orig_sent,
                    "back_translation": best_match_back_sent,
                    "similarity": max_similarity
                })
        return discrepancies

# -----------------------------
# Enhanced Critique Agent using Llama3 (for refinement)
# -----------------------------
class EnhancedCritiqueAgent:
    def __init__(self, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_critique_duration = 0
        self.total_critique_tokens = 0

    async def generate_async(self, prompt: str) -> tuple:
        return await async_generate(prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)

    def refine_with_discrepancies(self, translation: str, source_text: str, back_translation: str, discrepancies: list, src_lang: str, tgt_lang: str) -> str:
        discrepancy_text = ""
        if discrepancies:
            for i, disc in enumerate(discrepancies):
                discrepancy_text += f"Discrepancy {i+1}:\n- Original: {disc['original']}\n- Back Translation: {disc['back_translation']}\n- Similarity: {disc['similarity']:.2f}\n\n"
        else:
            discrepancy_text = "No significant discrepancies identified."

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
            improved_translation, duration, prompt_tokens, completion_tokens = asyncio.run(self.generate_async(prompt))
            self.total_critique_duration += duration
            self.total_critique_tokens += prompt_tokens + completion_tokens

            # Post-process: extract text after the marker, if present.
            if "Improved" in improved_translation:
                improved_translation = improved_translation.split("Improved", 1)[-1].strip()
            if "Explanation:" in improved_translation:
                improved_translation = improved_translation.split("Explanation:")[0].strip()
            if f"Improved {tgt_lang} Translation:" in improved_translation:
                 improved_translation = improved_translation.split(f"Improved {tgt_lang} Translation:", 1)[-1].strip()
            return improved_translation
        except Exception as e:
            print(f"Error during refinement: {e}")
            return translation # Return original translation in case of error

# -----------------------------
# Convergent Back-Translation Refinement Agent
# -----------------------------
class ConvergentBackTranslationImprover:
    def __init__(self, translator: BaseTranslator, critique_agent: EnhancedCritiqueAgent, similarity_calculator: SimilarityCalculator):
        self.translator = translator  # Instance of NLLBTranslator or MBARTTranslator
        self.critique_agent = critique_agent  # Llama3-based agent for refinement
        self.similarity_calculator = similarity_calculator

    def improve_translation(self, source_text: str, src_lang_name: str, tgt_lang_name: str, convergence_threshold: float = 0.90,
                            max_iterations: int = 5, min_improvement_delta: float = 0.01, debug: bool = True) -> dict:
        results = {
            "iterations": [],
            "original_text": source_text,
            "initial_translation": None,
            "final_translation": None,
            "convergence_achieved": False,
            "similarity_scores": [],
            "translation_history": [],
            "back_translations": [],
            "total_nmt_duration": 0, # Renamed for generality
            "total_nmt_tokens": 0,    # Renamed for generality
            "total_critique_duration": 0,
            "total_critique_tokens": 0,
        }
        
        if debug:
            print("===== ORIGINAL SOURCE TEXT =====")
            print(source_text)
            print("==================================\n")

        # Initial NMT (e.g., NLLB or mBART) translation
        primary_translation = self.translator.translate(source_text, src_lang_name, tgt_lang_name)
        results["initial_translation"] = primary_translation
        results["translation_history"].append(primary_translation)
        
        if debug:
            print(f"===== PRIMARY TRANSLATION ({self.translator.name}) =====")
            print(primary_translation)
            print("=================================\n")

        # Initial NMT back-translation
        back_translation = self.translator.translate(primary_translation, tgt_lang_name, src_lang_name)
        results["back_translations"].append(back_translation)
        
        if debug:
            print(f"===== INITIAL BACK TRANSLATION ({self.translator.name}) =====")
            print(back_translation if back_translation else "[Empty back translation]")
            print("=====================================\n")
        
        initial_similarity = self.similarity_calculator.calculate_similarity(source_text, back_translation)
        results["similarity_scores"].append(initial_similarity)
        print(f"Initial similarity: {initial_similarity:.4f}")

        current_translation = primary_translation
        current_similarity = initial_similarity
        
        # Accumulate initial translation/back-translation costs
        results["total_nmt_duration"] += self.translator.total_duration
        results["total_nmt_tokens"] += self.translator.total_tokens
        self.translator.total_duration = 0 # Reset for next call
        self.translator.total_tokens = 0 # Reset for next call

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
            
            if current_similarity >= convergence_threshold and not discrepancies: # Also check if no discrepancies
                print(f"Convergence achieved with similarity {current_similarity:.4f} >= {convergence_threshold} and no discrepancies.")
                results["convergence_achieved"] = True
                break
            
            previous_similarity = current_similarity
            
            # Refine using critique agent
            refined_translation = self.critique_agent.refine_with_discrepancies(
                current_translation, source_text, back_translation, discrepancies, src_lang_name, tgt_lang_name
            )
            print("\nRefined Translation (LLM-BTI):")
            print(refined_translation)
            
            results["total_critique_duration"] += self.critique_agent.total_critique_duration
            results["total_critique_tokens"] += self.critique_agent.total_critique_tokens
            self.critique_agent.total_critique_duration = 0 # Reset
            self.critique_agent.total_critique_tokens = 0 # Reset

            if refined_translation.strip() == current_translation.strip():
                print("No meaningful change detected by LLM. Ending iterations.\n")
                break
            
            current_translation = refined_translation
            results["translation_history"].append(current_translation)
            
            # Back-translate refined translation using NMT
            back_translation = self.translator.translate(current_translation, tgt_lang_name, src_lang_name)
            results["back_translations"].append(back_translation)
            
            results["total_nmt_duration"] += self.translator.total_duration
            results["total_nmt_tokens"] += self.translator.total_tokens
            self.translator.total_duration = 0 # Reset
            self.translator.total_tokens = 0 # Reset

            if debug:
                print(f"\nUpdated Back Translation ({self.translator.name}):")
                print(back_translation if back_translation else "[Empty back translation]")
            
            current_similarity = self.similarity_calculator.calculate_similarity(source_text, back_translation)
            results["similarity_scores"].append(current_similarity)
            print(f"\nNew similarity: {current_similarity:.4f}")
            iteration_data["similarity"] = current_similarity
            
            # Check for diminishing returns (improvement delta)
            improvement_delta = current_similarity - previous_similarity
            if improvement_delta < min_improvement_delta and improvement_delta >= 0: # Only if it's not a regression
                print(f"Improvement delta {improvement_delta:.4f} is below threshold {min_improvement_delta}. Stopping iterations.\n")
                break
            elif improvement_delta < 0 and iteration > 1: # If similarity decreases significantly after initial improvements
                print(f"Similarity decreased in this iteration ({improvement_delta:.4f}). Stopping to prevent degradation.\n")
                # Decide if you want to revert to the previous best translation here.
                # For now, we just stop.
                break 

            results["iterations"].append(iteration_data)

        results["final_translation"] = current_translation
        return results

# -----------------------------
# FloresTranslationEvaluator Class
# -----------------------------
class FloresTranslationEvaluator: 
    def __init__(self, dataset_name="openlanguagedata/flores_plus", split='devtest'):
        self.dataset_name = dataset_name
        self.split = split

        print(f"Loading dataset '{dataset_name}' split '{split}' (all languages)...")
        # Load the full multilingual dataset for the chosen split
        self.dataset = load_dataset(self.dataset_name, split=self.split)

        # Optional: Convert to Pandas for easier filtering (recommended for large datasets)
        import pandas as pd
        self.df = self.dataset.to_pandas()

        # Metrics unchanged
        self.comet_metric = evaluate.load("comet", config_name="wmt20-comet-da")
        self.chrf_metric = evaluate.load("chrf")
    def run_pipeline_on_example(self, index: int, 
                                nllb_translator: NLLBTranslator, 
                                mbart_translator: MBARTTranslator, 
                                critique_agent: EnhancedCritiqueAgent,
                                llm_only_translator: LLMOnlyTranslator, 
                                similarity_calculator_instance: SimilarityCalculator,
                                src_lang_code: str = "eng",   # iso_639_3
                                src_script: str = "Latn",
                                tgt_lang_code: str = "kor",   # iso_639_3
                                tgt_script: str = "Hang",
                                max_iterations: int = 3, 
                                debug: bool = True) -> dict:

        # Filter and align parallel sentences
        src_rows = self.df[
            (self.df['iso_639_3'] == src_lang_code) & 
            (self.df['iso_15924'] == src_script)
        ].sort_values('id').reset_index(drop=True)

        tgt_rows = self.df[
            (self.df['iso_639_3'] == tgt_lang_code) & 
            (self.df['iso_15924'] == tgt_script)
        ].sort_values('id').reset_index(drop=True)

        if index >= len(src_rows) or index >= len(tgt_rows):
            raise IndexError(f"Index {index} out of range for language pair {src_lang_code}-{tgt_lang_code}.")

        source_text = src_rows.loc[index, 'text']
        reference_translation = tgt_rows.loc[index, 'text']
        sentence_id = src_rows.loc[index, 'id']

        src_lang_name = get_nllb_lang_name(f"{src_lang_code}_{src_script}")
        tgt_lang_name = get_nllb_lang_name(f"{tgt_lang_code}_{tgt_script}")

        if debug:
            print(f"\nSource text ({src_lang_name}) from dataset (id={sentence_id}):")
            print(source_text)
            print(f"\nReference translation ({tgt_lang_name}) from dataset:")
            print(reference_translation)

        # Reset cumulative costs before each example
        nllb_translator.total_duration = 0
        nllb_translator.total_tokens = 0
        mbart_translator.total_duration = 0
        mbart_translator.total_tokens = 0
        critique_agent.total_critique_duration = 0
        critique_agent.total_critique_tokens = 0
        llm_only_translator.total_duration = 0
        llm_only_translator.total_tokens = 0

        # 1. LLM-BTI with NLLB
        print("\n--- Running LLM-BTI (NLLB as base) ---")
        llm_bti_nllb_improver = ConvergentBackTranslationImprover(
            nllb_translator, critique_agent, similarity_calculator_instance
        )
        llm_bti_nllb_results = llm_bti_nllb_improver.improve_translation(
            source_text, src_lang_name, tgt_lang_name,
            convergence_threshold=0.90, max_iterations=max_iterations,
            min_improvement_delta=0.005, debug=debug
        )
        final_translation_llm_bti_nllb = llm_bti_nllb_results["final_translation"]

        # 2. LLM-BTI with mBART
        print("\n--- Running LLM-BTI (mBART as base) ---")
        llm_bti_mbart_improver = ConvergentBackTranslationImprover(
            mbart_translator, critique_agent, similarity_calculator_instance
        )
        llm_bti_mbart_results = llm_bti_mbart_improver.improve_translation(
            source_text, src_lang_name, tgt_lang_name,
            convergence_threshold=0.90, max_iterations=max_iterations,
            min_improvement_delta=0.005, debug=debug
        )
        final_translation_llm_bti_mbart = llm_bti_mbart_results["final_translation"]

        # 3. NLLB Primary
        primary_translation_nllb = nllb_translator.translate(source_text, src_lang_name, tgt_lang_name)

        # 4. mBART Primary
        primary_translation_mbart = mbart_translator.translate(source_text, src_lang_name, tgt_lang_name)

        # 5. LLM-Only Direct
        llm_only_translation = llm_only_translator.translate(source_text, src_lang_name, tgt_lang_name)

        if debug:
            print(f"\nNLLB Primary: {primary_translation_nllb}")
            print(f"mBART Primary: {primary_translation_mbart}")
            print(f"LLM Only: {llm_only_translation}")
            print(f"LLM-BTI (NLLB): {final_translation_llm_bti_nllb}")
            print(f"LLM-BTI (mBART): {final_translation_llm_bti_mbart}")

        # Evaluation
        primary_nllb_comet = self.comet_metric.compute(
            sources=[source_text], predictions=[primary_translation_nllb], references=[reference_translation]
        )["scores"][0]
        primary_nllb_chrf = self.chrf_metric.compute(
            predictions=[primary_translation_nllb], references=[reference_translation]
        )["score"]

        primary_mbart_comet = self.comet_metric.compute(
            sources=[source_text], predictions=[primary_translation_mbart], references=[reference_translation]
        )["scores"][0]
        primary_mbart_chrf = self.chrf_metric.compute(
            predictions=[primary_translation_mbart], references=[reference_translation]
        )["score"]

        llm_only_comet = self.comet_metric.compute(
            sources=[source_text], predictions=[llm_only_translation], references=[reference_translation]
        )["scores"][0]
        llm_only_chrf = self.chrf_metric.compute(
            predictions=[llm_only_translation], references=[reference_translation]
        )["score"]

        final_llm_bti_nllb_comet = self.comet_metric.compute(
            sources=[source_text], predictions=[final_translation_llm_bti_nllb], references=[reference_translation]
        )["scores"][0]
        final_llm_bti_nllb_chrf = self.chrf_metric.compute(
            predictions=[final_translation_llm_bti_nllb], references=[reference_translation]
        )["score"]

        final_llm_bti_mbart_comet = self.comet_metric.compute(
            sources=[source_text], predictions=[final_translation_llm_bti_mbart], references=[reference_translation]
        )["scores"][0]
        final_llm_bti_mbart_chrf = self.chrf_metric.compute(
            predictions=[final_translation_llm_bti_mbart], references=[reference_translation]
        )["score"]

        return {
            "primary_nllb_comet": primary_nllb_comet, "primary_nllb_chrf": primary_nllb_chrf,
            "primary_mbart_comet": primary_mbart_comet, "primary_mbart_chrf": primary_mbart_chrf,
            "llm_only_comet": llm_only_comet, "llm_only_chrf": llm_only_chrf,
            "final_llm_bti_nllb_comet": final_llm_bti_nllb_comet, "final_llm_bti_nllb_chrf": final_llm_bti_nllb_chrf,
            "final_llm_bti_mbart_comet": final_llm_bti_mbart_comet, "final_llm_bti_mbart_chrf": final_llm_bti_mbart_chrf,

            "llm_bti_nllb_duration": llm_bti_nllb_results["total_nmt_duration"] + llm_bti_nllb_results["total_critique_duration"],
            "llm_bti_nllb_tokens": llm_bti_nllb_results["total_nmt_tokens"] + llm_bti_nllb_results["total_critique_tokens"],
            "llm_bti_mbart_duration": llm_bti_mbart_results["total_nmt_duration"] + llm_bti_mbart_results["total_critique_duration"],
            "llm_bti_mbart_tokens": llm_bti_mbart_results["total_nmt_tokens"] + llm_bti_mbart_results["total_critique_tokens"],
            "llm_only_duration": llm_only_translator.total_duration,
            "llm_only_tokens": llm_only_translator.total_tokens,
        }

    def run_pipeline_on_examples(self, src_code: str, tgt_code: str, indices: list,
                                 nllb_translator: NLLBTranslator, mbart_translator: MBARTTranslator,
                                 : EnhancedCritiqueAgent,
                                 llm_only_translator: LLMOnlyTranslator, 
                                 similarity_calculator_instance: SimilarityCalculator,
                                 max_iterations: int = 3, debug: bool = True):

        # Extract iso_639_3 and script from NLLB-style codes (e.g., "eng_Latn" → "eng", "Latn")
        src_lang_code, src_script = src_code.split("_")
        tgt_lang_code, tgt_script = tgt_code.split("_")

        scores = {
            "primary_nllb_comet": [], "primary_nllb_chrf": [],
            "primary_mbart_comet": [], "primary_mbart_chrf": [],
            "llm_only_comet": [], "llm_only_chrf": [],
            "final_llm_bti_nllb_comet": [], "final_llm_bti_nllb_chrf": [],
            "final_llm_bti_mbart_comet": [], "final_llm_bti_mbart_chrf": [],
        }
        computational_costs = {
            "llm_bti_nllb_duration": [], "llm_bti_nllb_tokens": [],
            "llm_bti_mbart_duration": [], "llm_bti_mbart_tokens": [],
            "llm_only_duration": [], "llm_only_tokens": []
        }

        for idx in indices:
            print(f"\n================== Running example {idx} ==================\n")
            try:
                result = self.run_pipeline_on_example(
                    idx, nllb_translator, mbart_translator, ,
                    llm_only_translator, similarity_calculator_instance,
                    src_lang_code=src_lang_code, src_script=src_script,
                    tgt_lang_code=tgt_lang_code, tgt_script=tgt_script,
                    max_iterations=max_iterations, debug=debug
                )

                # Append scores
                for key in scores:
                    scores[key].append(result[key])

                # Append costs
                computational_costs["llm_bti_nllb_duration"].append(result["llm_bti_nllb_duration"])
                computational_costs["llm_bti_nllb_tokens"].append(result["llm_bti_nllb_tokens"])
                computational_costs["llm_bti_mbart_duration"].append(result["llm_bti_mbart_duration"])
                computational_costs["llm_bti_mbart_tokens"].append(result["llm_bti_mbart_tokens"])
                computational_costs["llm_only_duration"].append(result["llm_only_duration"])
                computational_costs["llm_only_tokens"].append(result["llm_only_tokens"])

            except Exception as e:
                print(f"Error processing example {idx}: {e}")

        n = len(scores["final_llm_bti_nllb_comet"])
        if n == 0:
            print("No scores to average.")
            return

        # Compute averages and std
        metrics_summary = {}
        for key in scores:
            metrics_summary[key + "_avg"] = np.mean(scores[key])
            metrics_summary[key + "_std"] = np.std(scores[key])

        # Print results (same as your original)
        print(f"\n================== Average Evaluation Scores over {n} examples ==================")
        print(f"Avg NLLB Primary - COMET: {metrics_summary['primary_nllb_comet_avg']:.4f} (±{metrics_summary['primary_nllb_comet_std']:.4f}), chrF: {metrics_summary['primary_nllb_chrf_avg']:.4f}")
        print(f"Avg mBART Primary - COMET: {metrics_summary['primary_mbart_comet_avg']:.4f} (±{metrics_summary['primary_mbart_comet_std']:.4f}), chrF: {metrics_summary['primary_mbart_chrf_avg']:.4f}")
        print(f"Avg LLM Only Direct - COMET: {metrics_summary['llm_only_comet_avg']:.4f} (±{metrics_summary['llm_only_comet_std']:.4f}), chrF: {metrics_summary['llm_only_chrf_avg']:.4f}")
        print(f"Avg LLM-BTI (NLLB Base) - COMET: {metrics_summary['final_llm_bti_nllb_comet_avg']:.4f} (±{metrics_summary['final_llm_bti_nllb_comet_std']:.4f}), chrF: {metrics_summary['final_llm_bti_nllb_chrf_avg']:.4f}")
        print(f"Avg LLM-BTI (mBART Base) - COMET: {metrics_summary['final_llm_bti_mbart_comet_avg']:.4f} (±{metrics_summary['final_llm_bti_mbart_comet_std']:.4f}), chrF: {metrics_summary['final_llm_bti_mbart_chrf_avg']:.4f}")

        # Plotting code remains the same (you can keep your existing plotting section)

        # --- Plotting ---
        labels = ['COMET', 'chrF']
        
        # Data for each bar group
        # Using a list of tuples to keep method name and its scores together for easy iteration
        methods_data = [
            ("NLLB Primary", [metrics_summary['primary_nllb_comet_avg'], metrics_summary['primary_nllb_chrf_avg']], [metrics_summary['primary_nllb_comet_std'], metrics_summary['primary_nllb_chrf_std']]),
            ("mBART Primary", [metrics_summary['primary_mbart_comet_avg'], metrics_summary['primary_mbart_chrf_avg']], [metrics_summary['primary_mbart_comet_std'], metrics_summary['primary_mbart_chrf_std']]),
            ("LLM Only Direct", [metrics_summary['llm_only_comet_avg'], metrics_summary['llm_only_chrf_avg']], [metrics_summary['llm_only_comet_std'], metrics_summary['llm_only_chrf_std']]),
            ("LLM-BTI (NLLB Base)", [metrics_summary['final_llm_bti_nllb_comet_avg'], metrics_summary['final_llm_bti_nllb_chrf_avg']], [metrics_summary['final_llm_bti_nllb_comet_std'], metrics_summary['final_llm_bti_nllb_chrf_std']]),
            ("LLM-BTI (mBART Base)", [metrics_summary['final_llm_bti_mbart_comet_avg'], metrics_summary['final_llm_bti_mbart_chrf_avg']], [metrics_summary['final_llm_bti_mbart_comet_std'], metrics_summary['final_llm_bti_mbart_chrf_std']]),
        ]

        x = np.arange(len(labels))  # the label locations
        num_methods = len(methods_data)
        bar_width = 0.8 / num_methods # Adjust width based on number of methods
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot bars for each method
        for i, (method_name, avg_scores, std_scores) in enumerate(methods_data):
            offset = bar_width * (i - (num_methods - 1) / 2)
            rects = ax.bar(x + offset, avg_scores, bar_width, yerr=std_scores, capsize=5, label=method_name)
            
            # Add value labels on top of bars
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)

        ax.set_ylabel('Average Scores')
        ax.set_title(f'Average Translation Scores: {get_nllb_lang_name(src_code)} to {get_nllb_lang_name(tgt_code)} (N={n})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize='small') # Place legend outside
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.25) # Adjust subplot to make space for the legend
        plt.show()

# -----------------------------
# Main Execution: Evaluate on Dataset Examples
# -----------------------------
if __name__ == "__main__":
    # List of language pairs to evaluate (add as many as you want!)
    LANGUAGE_PAIRS = [
        ("eng_Latn", "kor_Hang"),    # English → Korean
        # ("eng_Latn", "jpn_Jpan"),    # English → Japanese
        # ("eng_Latn", "fra_Latn"),    # English → French
        # ("eng_Latn", "spa_Latn"),    # English → Spanish
        # ("eng_Latn", "hin_Deva"),    # English → Hindi (low-resource example)
        # # Add more: e.g., ("eng_Latn", "zho_Hans") for Chinese Simplified
    ]

    NUM_EXAMPLES_TO_EVALUATE = 5
    MAX_ITERATIONS_FOR_LLMBTI = 3
    SIMILARITY_CALCULATOR_TYPE = "SentenceTransformer"

    # --- Initialize Models (unchanged) ---
    nllb_translator = NLLBTranslator(get_code_func=get_nllb_code, max_length=512)
    mbart_translator = MBARTTranslator(get_code_func=get_mbart_code, max_length=512)
    critique_agent = EnhancedCritiqueAgent(model="llama-3.3-70b-versatile", max_tokens=512)
    llm_only_translator = LLMOnlyTranslator(model="llama-3.3-70b-versatile", max_tokens=512)

    if SIMILARITY_CALCULATOR_TYPE == "LCS":
        similarity_calculator = LCSBasedSimilarityCalculator()
        print("Using LCS-based similarity for discrepancy detection.")
    elif SIMILARITY_CALCULATOR_TYPE == "SentenceTransformer":
        similarity_calculator = SentenceTransformerSimilarityCalculator()
        print("Using Sentence-Transformer based similarity for discrepancy detection.")
    else:
        raise ValueError("Invalid SIMILARITY_CALCULATOR_TYPE.")

    # --- Create Evaluator Instance (unchanged) ---
    flores_evaluator = FloresTranslationEvaluator(split='devtest')

    # --- Loop Over Each Language Pair ---
    for src_code, tgt_code in LANGUAGE_PAIRS:
        print(f"\n\n===== Starting Evaluation for {get_nllb_lang_name(src_code)} → {get_nllb_lang_name(tgt_code)} =====")
        print(f"--- Base NMT for LLM-BTI: NLLB and mBART ---\n")

        # Get number of available examples for this pair
        actual_num_examples = len(flores_evaluator.df[
            (flores_evaluator.df['iso_639_3'] == src_code.split('_')[0]) &
            (flores_evaluator.df['iso_15924'] == src_code.split('_')[1])
        ])
        indices_to_use = list(range(min(actual_num_examples, NUM_EXAMPLES_TO_EVALUATE)))

        # Run the pipeline for this pair
        flores_evaluator.run_pipeline_on_examples(
            src_code, tgt_code,
            indices_to_use, nllb_translator, mbart_translator, critique_agent, llm_only_translator,
            similarity_calculator,
            max_iterations=MAX_ITERATIONS_FOR_LLMBTI, debug=True  # Set debug=True for verbose output
        )

        print(f"===== Completed Evaluation for {get_nllb_lang_name(src_code)} → {get_nllb_lang_name(tgt_code)} =====\n")

    print("All language pairs evaluated successfully!")

    # --- Create Evaluator Instance ---
    flores_evaluator = FloresTranslationEvaluator(split='devtest')  # 'devtest' has ~2000 sentences, better for evaluation
    
    actual_num_examples = len(flores_evaluator.dataset)
    indices_to_use = list(range(min(actual_num_examples, NUM_EXAMPLES_TO_EVALUATE)))
    print(f"--- Base NMT for LLM-BTI: NLLB and mBART ---")

    flores_evaluator.run_pipeline_on_examples(
        indices_to_use, nllb_translator, mbart_translator, critique_agent, llm_only_translator, 
        similarity_calculator, # Pass the instance here
        max_iterations=MAX_ITERATIONS_FOR_LLMBTI, debug=True # Set debug=True for detailed output per example
    )
