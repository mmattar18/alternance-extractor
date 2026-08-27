# Project Explanation: Alternance Extractor

## The Goal
The **alternance-extractor** project is designed to evaluate how much of the performance gap in structured data extraction between a small, open-source model and a large API model can be closed using QLoRA fine-tuning. Specifically, it tests **Qwen2.5-1.5B-Instruct** (fine-tuned using QLoRA on a free Kaggle T4) against a few-shot prompted baseline of itself and the **Llama-3.3-70B model via Groq**. The task is to extract structured JSON data from unstructured French job postings (specifically targeting apprenticeship/alternance roles).

The final results indicate that the fine-tuned 1.5B model achieves a **0.770 macro F1** score on the hand-corrected 100-posting test set, closing **73.7%** of the gap between the 1.5B base model (0.432 F1) and the 70B Groq baseline (0.891 F1). While it does not fully match the 70B model's performance, it is a significant improvement and even beats the 70B model on specific fields like `contract_type` and `duration_months`.

## The Pipeline
The project is organized into several distinct stages:

1. **Data Ingestion (`ingest/`)**:
   - The pipeline fetches real French job postings from the France Travail API using a predefined set of keyword searches (e.g., "data analyst alternance").
   - It performs immediate deduplication (by offer ID and text hash) and filters out postings with overly short descriptions.
   - PII (Personally Identifiable Information) stripping and basic text cleaning are also handled here to prepare the raw data for labeling.

2. **Schemas (`schema/`)**:
   - A locked Pydantic schema (`schema/posting.py`) defines the exact structure of the expected JSON output, including fields like `title`, `company`, `contract_type`, `skills`, and `salary_range`.
   - The schema enforces strict rules (e.g., optional fields are `null` if not mentioned, and `0` is distinct from `null` for minimum years of experience).

3. **Labeling and Hand-Correction (`label/`)**:
   - The raw data is initially labeled by Groq's Llama-3.3-70B using a few-shot prompt (`label/prompt.py`) that embeds the JSON schema.
   - A training set of 541 postings and a test set of 100 postings are created. The test set is entirely hand-corrected using a custom review server UI (`label/review_server.py`) to serve as the ground truth (gold standard).
   - Targeted cleanup passes were also applied to the training set to fix systematic errors made by Groq (e.g., leaving `duration_months` null when a range is given, or overgenerating `required_skills`).

4. **Training (`notebooks/`)**:
   - The QLoRA fine-tuning of Qwen2.5-1.5B-Instruct is performed via Jupyter notebooks designed to run on Kaggle (`notebooks/kaggle_train.ipynb`).
   - The model is trained on the 541 labeled postings. A critical optimization is the use of a shortened prompt (`SYSTEM_PROMPT_SHORT`) that removes the bulky 1173-token JSON schema dump, relying on the fine-tuning process to teach the model the output structure.

5. **Evaluation Scoring (`eval/`)**:
   - The evaluation harness (`eval/score.py`) computes field-level exact match and partial (per-item) F1 scores.
   - It incorporates specific text normalization logic (e.g., stripping department prefixes from locations, normalizing connectors like "ou"/"et", handling French elisions) to ensure models are penalized for genuine extraction errors rather than formatting or typographical differences.

## Results & Insights
- **Performance Gap**: Fine-tuning improved the 1.5B model from a baseline of 0.432 F1 to 0.770 F1, successfully closing about 73.7% of the gap to the 70B teacher model.
- **Noise Floor**: Analysis showed a training stochasticity of ~0.03 macro F1. Small differences below this threshold between different training configurations are indistinguishable from noise.
- **Training Bug Discovery**: A major finding during the project was that using a standard conversational dataset shape (`{"messages": [...]}`) resulted in computing the loss over the entire sequence (~94% of which was the constant system prompt and input text). Switching to a prompt-completion shape (`completion_only_loss=True`) drastically improved the model's ability to learn the actual task, fixing a severe abstention failure where the model hallucinated skills instead of returning `null`.
- **Labeling Challenges**: `required_skills` proved to be the hardest field for both models. Groq (the teacher) systematically overgenerated skills by including generic competencies and soft skills, capping its precision. Qwen (the student) struggled with the same issue, highlighting a distillation caveat where a student model cannot easily overcome the systematic flaws in its teacher's labels. Attempts to fix this with regex-based data cleanup failed, suggesting a true capability limit at 1.5B parameters for nuanced skill extraction without fully hand-corrected training data.
- **Prompt Truncation Benefit**: Fine-tuning allowed for the removal of the JSON schema from the prompt without any measurable loss in performance, proving that the short prompt is "free" and saves significant token overhead during inference.
