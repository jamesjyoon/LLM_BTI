
# LLM_BTI

**Enhancing Multilingual Translation Utilizing Back-Translation During Inference**

## Overview

This project implements advanced multilingual translation using a combination of LLM-based generation, back-translation, and iterative refinement. It utilizes models such as NLLB (`facebook/nllb-200-distilled-600M`) and Llama3 (via Portkey), and evaluates results using metrics like COMET and chrF.

## Features
- **Back-Translation during Inference** for robust evaluation and improvement of translations
- **Convergent iterative refinement** with LLM-based critique and correction
- Multiple translation models (NLLB, Llama3)
- **Evaluation using standard metrics**: COMET, chrF
- **Example pipelines and recipe for Flores200 dataset evaluation**

## Requirements

See `requirements.txt`; install with:
```
pip install -r requirements.txt
```

**Required Model/Service Keys:**  
Create a `.env` file with your keys for:
- `PORTKEY_API_KEY`
- `GROQ_VIRTUAL_KEY`
- `HUGGING_FACE_HUB_TOKEN`

## Usage

### Back-Translation Pipeline
Run the **main script** for translation, critique, and evaluation:
```
python LLM_BTI.py
```

For a **free prompt demo** with user input:
```
python LLM_BTI_free_prompt.py
```

### Evaluation Pipeline
The pipeline can be used to automatically evaluate the translation quality on the Flores200 dataset (or other supported datasets). See classes and methods in `LLM_BTI.py` for batch processing and result plotting.

## File Structure

- `LLM_BTI.py` — main script for dataset-based translation and evaluation
- `LLM_BTI_free_prompt.py` — console-based user prompt workflow

## Citation

If you use this code/project in your research, please cite
```
@software{jamesjyoon_LLM_BTI,
  author = {James Y. Yoon},
  title = {LLM_BTI: Enhancing Multilingual Translation Utilizing Back-Translation During Inference},
  year = {2025},
  url = {https://github.com/jamesjyoon/LLM_BTI}
}
```
```

***

Let me know if you want any customization to the requirements or the README structure!

[1](https://github.com/jamesjyoon/LLM_BTI)
