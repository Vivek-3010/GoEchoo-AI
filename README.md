# Go Echoo AI

A lightweight **post-ASR text cleanup and formatting model** for Go Echoo.

The model takes raw speech-to-text transcription and transforms it into clean, readable text by progressively learning **punctuation, capitalization, spelling correction, and intelligent formatting**.

The project uses a pretrained **T5-small** sequence-to-sequence Transformer and progressively fine-tunes it on task-specific Go Echoo datasets.

---

## Overview

Speech recognition systems primarily focus on converting speech into text. The resulting transcription can still contain issues such as:

* Missing punctuation
* Incorrect capitalization
* Spelling mistakes
* Poorly represented numbers and dates
* Unstructured information
* Lists that are difficult to read
* Corrections or backtracking in spoken language
* Technical terms that need proper formatting

Go Echoo AI works as a **post-processing layer after ASR**.

### High-Level Pipeline

```text
┌─────────────────────┐
│      User Speech    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Speech-to-Text / ASR│
└──────────┬──────────┘
           │
           ▼
      Raw Transcript
           │
           ▼
┌────────────────────────────┐
│       Go Echoo AI          │
│                            │
│  T5-small Encoder-Decoder  │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Cleaned / Formatted Text   │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│     Active Application     │
│   (Electron / Go Echoo)    │
└────────────────────────────┘
```

The current model development focuses on the **Go Echoo AI layer**, not the ASR system.

---

# Model Architecture

Go Echoo AI uses a pretrained **T5-small** model.

T5 stands for:

> **Text-to-Text Transfer Transformer**

The model represents the task as a sequence-to-sequence transformation:

```text
Raw / Noisy Text
       │
       ▼
   T5 Encoder
       │
       ▼
Contextual Representation
       │
       ▼
   T5 Decoder
       │
       ▼
Clean / Formatted Text
```

### Example

```text
Input:
hello my name is vivek and i have a meeting tomorrow at three pm

Output:
Hello, my name is Vivek, and I have a meeting tomorrow at 3 PM.
```

The model is **not trained from scratch**. A pretrained T5-small model is used as the starting point and then fine-tuned for Go Echoo's specific text transformation tasks.

---

# Why T5-small?

The task is naturally a **text-to-text transformation problem**:

```text
Input text → Output text
```

T5 is designed around this paradigm, making it suitable for tasks where the desired output is another text sequence.

T5-small was selected because it provides a relatively lightweight model for local experimentation and fine-tuning.

Current development is performed locally, including CPU-based training.

---

# Progressive Model Development

Instead of training separate independent models for every feature, Go Echoo AI uses **progressive fine-tuning**.

Each version starts from the previous version's trained checkpoint.

```text
                  T5-small
                     │
                     ▼
              ┌─────────────┐
              │     V1      │
              │ Punctuation │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │     V2      │
              │ + Capital.  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │     V3      │
              │ + Spelling  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │     V4      │
              │ + Formatting│
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │     V5      │
              │ + Fillers   │
              └─────────────┘
```

This approach allows new capabilities to be added while retaining the capabilities learned by previous versions.

---

# Version History

## V1 — Punctuation

The first version focuses on restoring punctuation to raw transcription.

### Capabilities

* Periods
* Commas
* Question marks
* Exclamation marks
* Basic sentence boundaries

### Example

```text
Input:
hello how are you

Output:
Hello, how are you?
```

---

## V2 — Capitalization

V2 builds on V1 and adds capitalization.

### Capabilities

* Sentence capitalization
* Proper-name capitalization
* Punctuation from V1

### Example

```text
Input:
hello my name is vivek

Output:
Hello, my name is Vivek.
```

---

## V3 — Spelling Correction

V3 builds on V2 and adds spelling correction.

### Capabilities

* Punctuation
* Capitalization
* Obvious spelling correction
* Technical-term capitalization/preservation

### Example

```text
Input:
hello my nam is vivek i am workng on a react projet

Output:
Hello, my name is Vivek. I am working on a React project.
```

---

## V4 — Intelligent Formatting

V4 extends the model beyond basic text cleanup.

The goal is to make spoken information more useful and readable while preserving the user's intended meaning.

### Current V4 focus

* Dates
* Times
* Names
* Prices
* Percentages
* URLs
* Email addresses
* Phone numbers
* Quantities
* Measurements
* Addresses
* Filenames
* Company and product names
* Technical terminology
* Numbered lists
* Bullet lists
* Action items
* Meeting/planning information
* Backtracking and corrections
* Mixed structured content

### Backtracking

Spoken language frequently contains corrections:

```text
Input:
let's meet at 2 actually let's make that 3 PM
```

Expected behavior:

```text
Let's meet at 3 PM.
```

The goal is to preserve the speaker's **final intended thought** rather than blindly reproducing the initial statement.

### Numbered Lists

Spoken ordered information can be converted into a numbered list.

```text
Input:
we need to do one update dependencies two fix authentication three deploy the application
```

Expected structure:

```text
1. Update dependencies
2. Fix authentication
3. Deploy the application
```

### Bullet Lists

Independent items can be represented as bullets when the context clearly indicates a list.

```text
Input:
for the project we need react postgres authentication and stripe
```

Expected structure:

```text
- React
- PostgreSQL
- Authentication
- Stripe
```

### Important Formatting Principle

The model should **not format everything**.

For example:

```text
I have two meetings today.
```

should remain a normal sentence rather than becoming a list.

Therefore the V4 dataset includes examples where formatting is appropriate as well as examples where formatting should **not** occur.

---

# Dataset

The model uses supervised input-target pairs.

Each example contains:

```json
{
  "input": "hello my name is vivek",
  "target": "Hello, my name is Vivek."
}
```

The `input` represents raw/noisy transcription.

The `target` represents the desired cleaned output.

Conceptually:

```text
Input
  ↓
Model
  ↓
Target
```

The model learns this transformation through supervised fine-tuning.

---

# Dataset Format

Datasets are stored as **JSONL** files.

Example:

```text
{"input":"hello how are you","target":"Hello, how are you?"}
{"input":"my name is vivek","target":"My name is Vivek."}
{"input":"i have a meeting tomorrow","target":"I have a meeting tomorrow."}
```

One JSON object is stored per line.

---

# Dataset Organization

```text
data/
├── raw/
│   ├── v1_dataset.jsonl
│   ├── v2_dataset.jsonl
│   ├── v3_dataset.jsonl
│   └── v4_dataset.jsonl
│
└── processed/
    ├── v1/
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    │
    ├── v2/
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    │
    ├── v3/
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    │
    └── v4/
        ├── train.jsonl
        ├── validation.jsonl
        └── test.jsonl
```

Raw and processed datasets are excluded from Git because of their size and project workflow.

---

# Data Preparation

The dataset preparation pipeline:

```text
Raw JSONL
    │
    ▼
Load examples
    │
    ▼
Validate JSON structure
    │
    ▼
Remove exact duplicates
    │
    ▼
Shuffle with fixed seed
    │
    ▼
80% Training
10% Validation
10% Test
```

A fixed random seed is used so that the dataset split is reproducible.

---

# Training / Validation / Test

The dataset is divided into three parts.

### Training Set

Used by the model to learn the transformation.

```text
Training → Model learns
```

### Validation Set

Used during training to monitor performance on unseen examples.

```text
Validation → Development monitoring
```

### Test Set

Kept separate from training and used for final evaluation.

```text
Test → Final performance measurement
```

The current split is approximately:

```text
80% → Training
10% → Validation
10% → Test
```

---

# Training Pipeline

The main training script is:

```text
src/train.py
```

Run:

```bash
python src/train.py --version v1
```

or:

```bash
python src/train.py --version v3
```

For V4:

```bash
python src/train.py --version v4
```

---

# Training Flow

```text
Version argument
      │
      ▼
Determine model lineage
      │
      ▼
Load previous model / T5-small
      │
      ▼
Load training + validation datasets
      │
      ▼
Tokenize input and target
      │
      ▼
Create sequence-to-sequence batches
      │
      ▼
Fine-tune T5
      │
      ├───────────────┐
      │               │
      ▼               ▼
Training          Validation
      │               │
      └───────┬───────┘
              ▼
        Save checkpoint
              │
              ▼
      New Go Echoo model
```

---

# Model Lineage in Code

The training script automatically determines the starting model.

For V1:

```text
T5-small
   ↓
Go Echoo V1
```

For V2:

```text
Go Echoo V1
   ↓
Go Echoo V2
```

For V3:

```text
Go Echoo V2
   ↓
Go Echoo V3
```

For V4:

```text
Go Echoo V3
   ↓
Go Echoo V4
```

This allows the project to progressively fine-tune the same model lineage.

---

# Tokenization

Before text can be processed by T5, it must be converted into tokens.

```text
"Hello, my name is Vivek."
             │
             ▼
         Tokenizer
             │
             ▼
        Token IDs
```

The tokenizer is loaded from the same model checkpoint being used for training.

Current maximum sequence length:

```text
128 tokens
```

Longer sequences are currently truncated.

---

# Task Prefix

The current training and inference pipeline uses the prefix:

```text
punctuate:
```

For example:

```text
punctuate: hello my name is vivek
```

This originated from the initial punctuation-restoration task and is currently retained throughout the progressive model versions.

As the task expands beyond punctuation, a more general task prefix such as `clean:` or `format:` can be evaluated in future iterations.

---

# Training Configuration

Current training configuration includes:

```text
Epochs                  : 3
Training batch size     : 4
Evaluation batch size   : 4
Learning rate           : 1e-4
Weight decay            : 0.01
Maximum sequence length : 128
Evaluation              : Every epoch
Checkpoint saving       : Every epoch
CPU training            : Enabled
Beam search             : 4 beams during generation
```

These values are currently chosen for local experimentation and can be adjusted based on model performance and available hardware.

---

# What Happens During Training?

For every batch:

```text
Input text
    │
    ▼
Tokenizer
    │
    ▼
Token IDs
    │
    ▼
T5 Encoder
    │
    ▼
T5 Decoder
    │
    ▼
Predicted output
    │
    ▼
Compare with target
    │
    ▼
Calculate loss
    │
    ▼
Backpropagation
    │
    ▼
Update model weights
```

This process repeats across batches and epochs.

---

# Loss

Loss measures how different the model's prediction is from the expected target during training.

The training process attempts to minimize this loss by updating the model's parameters.

A lower training loss generally indicates that the model is fitting the training objective better, but loss alone does not establish real-world performance.

Validation and test performance are therefore also important.

---

# Inference

The inference script is:

```text
src/inference.py
```

Run:

```bash
python src/inference.py --version v3
```

or:

```bash
python src/inference.py --version v4
```

The script starts an interactive CLI.

Example:

```text
Go Echoo V3 is ready!
Type 'exit' to quit.

You: hello my name is vivek
Go Echoo: Hello, my name is Vivek.
```

---

# Inference Pipeline

```text
User Input
    │
    ▼
Task Prefix
    │
    ▼
Tokenizer
    │
    ▼
PyTorch Tensor
    │
    ▼
T5 Model
    │
    ▼
Beam Search
    │
    ▼
Generated Token IDs
    │
    ▼
Tokenizer Decode
    │
    ▼
Clean Text
```

Inference uses:

```python
torch.no_grad()
```

because the model is not being trained during inference and therefore does not need gradient calculations.

---

# Beam Search

The current generation configuration uses:

```text
num_beams = 4
```

Beam search keeps multiple candidate sequences during generation instead of immediately committing to a single token sequence at every step.

Conceptually:

```text
                 Input
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Candidate A Candidate B Candidate C
        │          │          │
        └──────────┼──────────┘
                   ▼
             Best sequence
```

This can improve sequence generation quality, at the cost of additional computation.

---

# Evaluation

The evaluation script is:

```text
src/evaluate.py
```

Run:

```bash
python src/evaluate.py --version v3
```

For V4:

```bash
python src/evaluate.py --version v4
```

---

# Evaluation Pipeline

```text
Trained Model
     │
     ▼
Held-out Test Dataset
     │
     ▼
Generate prediction
     │
     ▼
Compare with expected target
     │
     ▼
Count exact matches
     │
     ▼
Calculate accuracy
```

The current evaluation metric is **exact-match accuracy**.

An example is counted as correct only when:

```text
Prediction == Expected Target
```

after trimming leading/trailing whitespace.

---

# Example Evaluation

If:

```text
Correct = 498
Total   = 500
```

then:

```text
Accuracy = (498 / 500) × 100
         = 99.60%
```

V1 achieved:

```text
498 / 500
99.60% exact-match accuracy
```

on its held-out test set.

Exact-match accuracy should not be interpreted as general language understanding accuracy.

---

# Why V4 Requires Additional Evaluation

Formatting introduces more possible variations than simple punctuation or spelling correction.

For example, two outputs could communicate the same information while having slightly different formatting.

Therefore V4 should be evaluated using both:

### Quantitative evaluation

* Exact-match accuracy
* Additional formatting-specific metrics where appropriate

### Qualitative evaluation

Manual tests covering:

* Dates
* Times
* Names
* Prices
* URLs
* Emails
* Phone numbers
* Lists
* Backtracking
* Technical terms
* Mixed structured content
* Cases where formatting should not occur

---

# Project Structure

```text
goechoo-ai/
│
├── data/
│   ├── raw/
│   │   ├── v1_dataset.jsonl
│   │   ├── v2_dataset.jsonl
│   │   ├── v3_dataset.jsonl
│   │   └── v4_dataset.jsonl
│   │
│   └── processed/
│       ├── v1/
│       │   ├── train.jsonl
│       │   ├── validation.jsonl
│       │   └── test.jsonl
│       │
│       ├── v2/
│       ├── v3/
│       └── v4/
│
├── models/
│   ├── goechoo-v1/
│   ├── goechoo-v2/
│   ├── goechoo-v3/
│   └── goechoo-v4/
│
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Core Files

## `prepare_dataset.py`

Responsible for preparing raw datasets.

Main responsibilities:

* Load JSONL
* Validate basic data structure
* Remove exact duplicates
* Shuffle examples
* Split into training, validation and test sets
* Save processed datasets

---

## `train.py`

Responsible for model fine-tuning.

Main responsibilities:

* Select model version
* Determine model lineage
* Load previous checkpoint
* Load datasets
* Tokenize inputs and targets
* Configure training
* Fine-tune the model
* Evaluate on validation data
* Save the trained model and tokenizer

---

## `evaluate.py`

Responsible for automated model evaluation.

Main responsibilities:

* Load trained model
* Load test dataset
* Generate predictions
* Compare predictions with expected targets
* Calculate exact-match accuracy

---

## `inference.py`

Responsible for interactive model testing.

Main responsibilities:

* Load a selected model
* Accept user input
* Generate cleaned text
* Display the output

---

# Technologies

| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| Python                    | ML development                  |
| PyTorch                   | Deep learning framework         |
| Hugging Face Transformers | T5 model and training utilities |
| Hugging Face Datasets     | Dataset management              |
| SentencePiece             | T5 tokenization support         |
| Accelerate                | Training/inference support      |
| Evaluate                  | Evaluation tooling              |

---

# Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

## Prepare a Dataset

```bash
python src/prepare_dataset.py --version v4
```

---

## Train a Model

```bash
python src/train.py --version v4
```

The model will be saved to:

```text
models/goechoo-v4/
```

---

## Evaluate a Model

```bash
python src/evaluate.py --version v4
```

---

## Run Interactive Inference

```bash
python src/inference.py --version v4
```

Type:

```text
exit
```

to stop the program.

---

# Reproducibility

The dataset preparation process uses:

```python
random.seed(42)
```

This ensures that the random shuffle and resulting dataset split can be reproduced.

Model versions are stored separately:

```text
goechoo-v1
goechoo-v2
goechoo-v3
goechoo-v4
```

This makes it possible to compare different stages of the model development process.

---

# Current Development Status

| Version | Capability                              | Status      |
| ------- | --------------------------------------- | ----------- |
| V1      | Punctuation                             | Completed   |
| V2      | + Capitalization                        | Completed   |
| V3      | + Spelling correction                   | Completed   |
| V4      | + Intelligent formatting                | In progress |
| V5      | + Filler removal                        | Planned     |
| V6      | Future improvements based on evaluation | Planned     |

---

# Current Limitations

### 1. Model Size

T5-small is intentionally lightweight, but its capacity is lower than larger language models.

### 2. Maximum Sequence Length

The current pipeline uses a maximum length of 128 tokens.

Longer inputs are currently truncated.

A future implementation may require chunking or a longer-context strategy for extended dictation.

### 3. Local CPU Training

Current experimentation is performed locally using CPU, which limits training speed.

### 4. Exact-Match Evaluation

Exact-match accuracy can be overly strict for formatting tasks.

Additional evaluation methods are required for V4 and later versions.

### 5. Progressive Fine-Tuning

Starting each version from the previous checkpoint helps preserve earlier capabilities, but it does not guarantee that no previous capability will degrade.

Regression testing across previous capabilities is therefore important.

---

# Future Development

Potential future iterations include:

* Filler-word removal
* Improved handling of spoken corrections
* Better formatting decisions
* More robust technical vocabulary handling
* Longer-context processing
* Improved evaluation metrics
* Robustness testing against noisy ASR output
* Multilingual fine-tuning and evaluation
* Optimization for production inference
* Integration with the Go Echoo Electron application

Future capabilities will be prioritized based on observed model failure cases rather than simply adding features without evaluation.

---

# Development Philosophy

The model is being developed incrementally.

```text
Build
  ↓
Train
  ↓
Evaluate
  ↓
Identify failures
  ↓
Improve dataset/model
  ↓
Train next version
  ↓
Regression test previous capabilities
  ↓
Repeat
```

The goal is not simply to achieve a high training score, but to build a model that reliably transforms real-world speech transcription into useful written text.

---

# Model Development Summary

Go Echoo AI currently follows this architecture:

```text
                PRETRAINED T5-SMALL
                        │
                        ▼
              ┌───────────────────┐
              │       V1          │
              │   Punctuation     │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │       V2          │
              │ + Capitalization  │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │       V3          │
              │ + Spelling        │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │       V4          │
              │ + Formatting      │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │       V5          │
              │ + Fillers         │
              └───────────────────┘
```

Each stage is evaluated independently while also checking whether previously learned capabilities remain intact.

---

## Summary

Go Echoo AI is a **task-specific text transformation model** built by progressively fine-tuning a pretrained T5-small model.

The current progression is:

```text
V1 → Punctuation
V2 → Capitalization
V3 → Spelling
V4 → Intelligent Formatting
V5 → Filler Removal
```

The ultimate objective is to provide Go Echoo with a lightweight post-ASR intelligence layer that transforms raw speech transcription into **clean, natural and contextually useful written text**.
