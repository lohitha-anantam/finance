# Finance STAIR – Structure-Aware Financial Report Assistant

This is a research starter implementation of the STAIR pipeline for financial annual reports and SEC 10-K filings.

## Important distinction
- **Training corpus**: Kaggle documents used to create ToCs and synthetic question → gold leaf-section examples, then LoRA-fine-tune the retriever.
- **Test document**: a new report uploaded to Streamlit. It does not need to be part of the Kaggle training corpus.
- The app's embedding fallback is only a baseline for testing. It is not the trained STAIR model. For the full STAIR experiment, train the adapter and set `STAIR_ADAPTER_PATH`.

## 1. Choose a Kaggle dataset
Choose a dataset containing full-length annual reports / 10-K / financial disclosure documents in PDF, TXT, Markdown, HTML or DOCX. A dataset containing only numerical CSV rows (e.g. credit-card transactions) is not suitable for the original STAIR ToC-based method because it has no long-document hierarchy.

Kaggle dataset discovery:
https://www.kaggle.com/datasets?search=annual%20reports%2010-k

Example dataset slug is supplied by you after selecting a dataset; do not assume all Kaggle finance datasets contain annual-report documents. The SEC 10-K corpus is also available from EDGAR sources, but verify licensing and size before use.

## 2. Install
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure credentials
Copy `.env.example` to `.env` and set:
```env
KAGGLE_API_TOKEN=your_new_kaggle_api_token
KAGGLE_DATASET=owner/dataset-slug
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```
Never commit `.env` or tokens to GitHub. New Kaggle API tokens use `KAGGLE_API_TOKEN`; KaggleHub version/authentication behavior can change, so update `kagglehub` if authentication fails.

## 4. Download and prepare training corpus
```bash
python dataset_loader.py --dataset owner/dataset-slug
python make_corpus.py
```
The parser handles text-based PDF, TXT, MD, HTML, DOCX and CSV. Scanned PDFs require OCR. CSV is treated as text records; it does not magically create report structure.

## 5. Generate synthetic QA training examples
```bash
python generate_training_data.py --limit 500
```
Review `data/stair_train.jsonl`. Check that every `gold_leaf` is an exact section title in that document's ToC and that the question is answerable from its section. Synthetic labels are noisy unless manually reviewed. For robust research, create train/validation/test splits by company or document, not random questions from the same report.

## 6. Train LoRA adapter (GPU recommended)
```bash
python train_stair.py --epochs 1
```
This starter uses Mistral-7B-Instruct-v0.2 with 4-bit loading and LoRA. It needs a compatible CUDA GPU and sufficient VRAM. The paper's longer context setting may exceed free Colab GPU memory; this implementation caps training sequences at 4096 tokens. It is a practical adaptation, not an exact reproduction of every paper hyperparameter.

After training, adapter is saved to `artifacts/stair-finance-lora`.

## 7. Run Streamlit
```bash
streamlit run app.py
```
Upload a fresh annual report PDF and ask a question. The app extracts headings, predicts a section, retrieves evidence and generates a grounded answer with OpenRouter.

## 8. Streamlit Cloud secrets
```toml
OPENROUTER_API_KEY = "..."
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```
For inference, Kaggle credentials are not required. Add `KAGGLE_API_TOKEN` only if you intend to download the training corpus from the deployed app (not recommended). Upload the trained adapter to private storage or Hugging Face Hub and download it during deployment; do not put a multi-GB model in GitHub.

## 9. Limitations / responsible use
- ToC headings are heuristically inferred. Verify against each original report; never label inferred headings as official report headings.
- PDF page labels are preserved when extractable as `[PAGE n]`; HTML/TXT sources may not have page numbers.
- Scanned PDFs need OCR.
- Financial answers are informational, not investment advice. Always verify figures against the original filing.
- The package has not been trained or live-tested on your chosen Kaggle corpus. First run dataset inspection and review the generated corpus.
