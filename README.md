---
language:
- en
- zh
- ja
- de
- fr
- es
- ar
- sw
tags:
- translation
- blind-spots
- evaluation
- error-analysis
- multilingual
pretty_name: HY-MT1.5-1.8B Blind Spot Probe Dataset
size_categories:
- n<100
---

# 🔍 HY-MT1.5-1.8B Blind Spot Probe Dataset

## Model Tested

**[tencent/HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B)**  
A 1.8-billion parameter neural machine translation model developed by Tencent Hunyuan.
It supports 33 languages and is optimised for Chinese↔Foreign and English↔Foreign translation pairs.
It performs strongly on clean, neutral benchmark text (FLORES-200) but was probed here
on naturalistic, culturally complex, and linguistically ambiguous inputs.

---

## How the Model Was Loaded

The model was loaded using Hugging Face `transformers` with `bfloat16` precision
on a Google Colab T4 GPU runtime.
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tencent/HY-MT1.5-1.8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
```

Translation was performed using the official chat-template prompt format:
```python
def translate(text: str, target_language: str, max_new_tokens: int = 512) -> str:
    prompt = (
        f"Translate the following segment into {target_language}, "
        f"without additional explanation.\n\n{text}"
    )
    messages = [{"role": "user", "content": prompt}]

    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = tokenized.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.6,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

**Environment:**
- Python 3.10
- `transformers==4.56.0`
- `torch==2.3.0`
- Google Colab, NVIDIA Tesla T4 GPU (16GB VRAM)

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique probe identifier (probe_01 … probe_12) |
| `category` | string | Blind spot category (e.g., "Idiomatic expression") |
| `source_language` | string | Language of the input text |
| `target_language` | string | Language the model was asked to translate into |
| `input_text` | string | The exact text fed to the model |
| `expected_output` | string | Correct human-verified or reference translation |
| `model_output` | string | The model's actual output |
| `error_type` | string | High-level classification of the expected error |
| `explanation` | string | Why this input was expected to cause a failure |
| `is_error` | bool | True if the model made a confirmed error |
| `error_note` | string | Short description of the specific error observed |
| `model_name` | string | Always `tencent/HY-MT1.5-1.8B` |

---

## Summary of Blind Spots Found

| ID | Category | Direction | Error Type |
|----|----------|-----------|------------|
| probe_01 | Idiomatic expression | EN → ZH | Literal translation of figurative idiom |
| probe_02 | Classical idiom (chengyu) | ZH → EN | Cultural reference lost in translation |
| probe_03 | Code-switching (Spanglish) | ES+EN → EN | Mixed-language input not fully handled |
| probe_04 | Internet slang / Gen-Z | EN → FR | Unknown evolving vocabulary |
| probe_05 | Long-text completeness | EN → ZH | Truncation beyond ~150 characters |
| probe_06 | Double negation | EN → ZH | Semantic polarity inversion |
| probe_07 | Low-resource language (Swahili) | SW → EN | Hallucination due to sparse training data |
| probe_08 | Medical domain jargon | EN → ZH | Non-standard terminology without glossary |
| probe_09 | Sarcasm and irony | EN → DE | Pragmatic register flattened |
| probe_10 | Financial number formatting | EN → JA | Japanese 億-unit notation not applied |
| probe_11 | Gender-ambiguous pronoun | EN → ES | Singular 'they' arbitrarily gendered |
| probe_12 | Arabic philosophical text | AR → EN | Abstract morphology mistranslated |

---

## What Fine-Tuning Data Would Fix These Errors?

The errors fall into **5 root causes**, each requiring a different type of fine-tuning data:

### 1. Idiomatic & Cultural Expressions (probes 01, 02, 09)
**Problem:** The model translates word-for-word instead of meaning-for-meaning.  
**Fix:** Fine-tune on a dataset of idiom-aligned pairs where the source contains
a figurative expression and the target contains its natural-language equivalent
in the target language — NOT a literal translation.

**Where to find it:**
- [EPIE Corpus](https://github.com/prateeksaxena2809/EPIE_Corpus) — English idiomatic expressions with literal vs. idiomatic translations
- Wiktionary idiom pages scraped and aligned across languages
- Human-translated literary fiction (novels), where translators routinely
  adapt idioms rather than translate them literally (e.g., Project Gutenberg parallel texts)

**How to assemble it:** Scrape idiom dictionaries in 10+ languages, align
source idiom → target equivalent using bilingual human translators or a stronger
model (GPT-4) as a first pass, then human-verify.

**Dataset size needed:** ~50,000–100,000 idiom pairs across the 33 supported languages
(roughly 1,500–3,000 pairs per language direction).

---

### 2. Code-Switching & Slang (probes 03, 04)
**Problem:** The model assumes each input is monolingual and has no knowledge
of rapidly evolving internet vocabulary.

**Fix:** Fine-tune on:
- **Code-switching corpora** — sentences that genuinely mix two languages
  (e.g., Spanglish, Hinglish, Franglais) paired with clean target-language translations
- **Slang/neologism translation pairs** — internet slang with dated timestamps
  so the model can be periodically refreshed

**Where to find it:**
- [LinCE Benchmark](https://ritual.uh.edu/lince/) — code-switching NLP benchmark with Spanish-English data
- Twitter/X and Reddit corpora filtered for code-switched posts (using a language-ID classifier)
- Urban Dictionary definitions scraped and aligned with formal equivalents in target languages

**How to assemble it:** Use a language identification model (e.g., `langdetect` or
`fastText`) to flag sentences containing tokens from 2+ languages. Then use
bilingual human annotators to write clean target-language translations.

**Dataset size needed:** ~20,000–50,000 examples per major code-switching pair
(EN-ES, EN-FR, EN-ZH). Slang data needs continuous updates — treat it as a
living dataset refreshed quarterly.

---

### 3. Long Text & Structural Completeness (probe 05)
**Problem:** The model drops the tail of long inputs, likely due to attention
degradation or training on predominantly short segments.

**Fix:** Fine-tune specifically on **long-document translation pairs** where
the model is rewarded for completeness — i.e., add a length-penalty term to
the training loss that penalises outputs significantly shorter than expected.

**Where to find it:**
- [WMT news translation tasks](https://www.statmt.org/wmt24/) — contains full
  article-length parallel texts in 10+ language pairs
- UN parallel corpus (long formal documents)
- EU legislation (EUR-Lex) — extremely long, formally translated documents
  in 24 languages

**How to assemble it:** Filter existing parallel corpora to keep only
document pairs where the source is >200 words. Crucially, add a completeness
check: verify the target word count is proportional to the source.

**Dataset size needed:** ~10,000–30,000 long-document pairs. Fewer examples
are needed here because the issue is structural (attention span) not vocabulary.

---

### 4. Domain-Specific Terminology (probe 08, 10)
**Problem:** Without a terminology glossary, the model uses non-standard or
incorrect domain-specific terms in medicine, finance, and law.

**Fix:** Fine-tune on **in-domain parallel corpora** with verified terminology:
- Medical: clinical notes, drug labels, medical journal abstracts
- Financial: earnings reports, SEC filings, financial news
- Legal: contracts, court documents

**Where to find it:**
- [UFAL Medical Corpus](https://ufal.mff.cuni.cz/ufal_medical_corpus) — 3M sentence pairs in medical domain
- [MultiUN](https://conferences.unite.un.org/uncorpus) — UN documents with legal/political terminology
- PubMed abstracts with Chinese/Japanese/German translations
- SEC EDGAR filings with parallel Japanese translations (for Japanese financial formatting)

**How to assemble it:** Download domain corpora, align at sentence level
using a tool like `hunalign`, then filter with a terminology consistency
checker (verify that known terms like "STEMI" always map to the same target term).

**Dataset size needed:** ~100,000–500,000 sentence pairs per domain.
Domain fine-tuning is data-hungry because the model must memorise a large
specialist vocabulary.

---

### 5. Pragmatics, Gender & Morphology (probes 06, 07, 09, 11, 12)
**Problem:** Semantic polarity errors (double negation), gender ambiguity,
sarcasm register loss, and Arabic morphological complexity are all
**pragmatic and grammatical** failures that require deeper linguistic signal.

**Fix:**
- For **gender ambiguity**: fine-tune on examples where the source explicitly
  flags ambiguity (e.g., via a comment or bracket) and the target uses a
  gender-neutral construction or the most contextually appropriate gender.
- For **sarcasm**: fine-tune on labelled sarcasm corpora where the ironic
  register is preserved in the translation.
- For **double negation**: augment training data with contrastive pairs
  that isolate negation structures.
- For **low-resource languages**: gather more Swahili, Yoruba, and other
  African language parallel data.

**Where to find it:**
- [MuST-SHE](https://mt.fbk.eu/must-she/) — gender bias in speech translation
- [SemEval sarcasm datasets](https://semeval.github.io/) — labelled sarcasm in multiple languages
- [OPUS corpus](https://opus.nlpl.eu/) — massive multilingual parallel corpus including Swahili
- [Masakhane](https://www.masakhane.io/) — community-built NLP datasets for African languages

**Dataset size needed:**
- Gender: ~10,000–20,000 targeted examples
- Sarcasm: ~5,000–10,000 (rare phenomenon, high quality > high quantity)
- Low-resource (Swahili): as much as possible — even 50,000 high-quality
  pairs would meaningfully reduce hallucination

---

## Overall Fine-Tuning Size Estimate

| Error Category | Recommended Dataset Size |
|----------------|--------------------------|
| Idioms & cultural expressions | 50,000–100,000 pairs |
| Code-switching & slang | 20,000–50,000 pairs (refresh quarterly) |
| Long-document completeness | 10,000–30,000 pairs |
| Domain jargon (medical, financial) | 100,000–500,000 pairs |
| Pragmatics, gender, morphology | 5,000–20,000 pairs |
| **Total (combined fine-tune mix)** | **~200,000–700,000 pairs** |

A combined fine-tuning dataset of around **300,000–500,000 high-quality,
diverse parallel sentence pairs** — carefully balanced across error categories
and language directions — would be a practical target. Quality matters more
than quantity: 100,000 clean, hard examples will outperform 1,000,000
noisy web-scraped pairs for fixing specific failure modes.

---

## Citation

If you use this dataset, please cite the original model:
```
@misc{hunyuan-mt-2025,
  title  = {HY-MT1.5: Tencent Hunyuan Machine Translation Model},
  author = {Tencent Hunyuan Team},
  year   = {2025},
  url    = {https://huggingface.co/tencent/HY-MT1.5-1.8B}
}
```

# HY-MT1.5-1.8B Blindspots Dataset

Dataset hosted on Hugging Face:
https://huggingface.co/datasets/Jesujuwon/HY-MT1.5-1.8B-blindspots

## Description
This dataset explores blindspots in the HY-MT1.5-1.8B model.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Jesujuwon/HY-MT1.5-1.8B-blindspots")
