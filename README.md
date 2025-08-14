# Sentiment Analysis of IMDB Movie Reviews using Hugging Face Transformers 🎬

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30%2B-yellow.svg)
![Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-2.14%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project is an end-to-end demonstration of fine-tuning a pretrained Transformer model for a text classification task. I use the `distilbert-base-uncased` model to perform sentiment analysis on the IMDB movie review dataset, classifying reviews as either **positive** or **negative**.

The entire workflow, from data loading to model deployment, is containerized in the main Jupyter Notebook.

## Project Objective

The goal is to build a reliable sentiment classifier by fine-tuning a modern NLP model and deploying it for inference on the Hugging Face Hub. This serves as a practical example of applying transfer learning to solve a common real-world problem.

---

## Features

* **Data Loading & Preprocessing:** Loads the `imdb` dataset directly from the Hugging Face Hub and tokenizes it using the `distilbert` tokenizer.
* **Model Fine-Tuning:** Fine-tunes the `distilbert-base-uncased` model for sequence classification using the high-level `Trainer` API from the `transformers` library.
* **Evaluation:** Measures model performance using standard classification metrics: accuracy, F1-score, precision, and recall. Visualizes results with a confusion matrix.
* **Inference:** Provides a simple function and `pipeline` to test the model on new, unseen text.
* **Deployment:** The final fine-tuned model is saved and pushed to the Hugging Face Hub for easy access and deployment.

---

## Deployed Model on Hugging Face Hub

The final model is publicly available on the Hugging Face Hub. You can use it for inference directly in your own projects with just a few lines of code.

**Model Hub Link:** [**timkilikimtai/distilbert-imdb-sentiment-analysis**](https://huggingface.co/timkilikimtai/distilbert-imdb-sentiment-analysis)

### Quick Inference

To use the model, install the `transformers` library (`pip install transformers`) and run the following Python code:

```python
from transformers import pipeline

# Load the sentiment analysis pipeline with our fine-tuned model
pipe = pipeline("sentiment-analysis", model="timothykimutai/distilbert-imdb-sentiment-analysis")

# Example 1: Positive review
result_pos = pipe("This movie was an absolute masterpiece! The acting was superb.")
print(result_pos)
# Output: [{'label': 'POSITIVE', 'score': 0.99...}]

# Example 2: Negative review
result_neg = pipe("I was really disappointed. The story was predictable and boring.")
print(result_neg)
# Output: [{'label': 'NEGATIVE', 'score': 0.99...}]
```

---

## Getting Started

To run this project locally, follow these steps.

### Prerequisites

* Python 3.8 or higher
* Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/timothykimutai/huggingface-sentiment-analysis.git](https://github.com/timothykimutai/huggingface-sentiment-analysis.git)
    cd huggingface-sentiment-analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Once the setup is complete, you can run the project:

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  Open the main notebook file (e.g., `sentiment_analysis.ipynb`).
3.  Run the cells from top to bottom to execute the entire data science workflow.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
````
