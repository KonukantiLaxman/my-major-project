🚨 Deep Learning Neural Network for Cyberbullying Detection in Social Media:

Automatically detect cyberbullying in text (tweets, comments, posts) using deep learning. Built for fast, production-ready inference and easy experimentation.

✨ Preview:
https://konukantilaxman.github.io/my-major-project/

🔎 What this project does

-Classifies text as bullying vs non-bullying (extendable to multi-class types like insult, threat, hate, etc.)

-Provides two model options:

-BiLSTM + Attention (lightweight, fast, trainable on CPU/GPU)

-Transformer (BERT) variant (higher accuracy, needs GPU for best speed)

-Includes clean training pipeline, evaluation metrics, confusion matrix, PR/ROC curves, and explainability (word-level attention or SHAP for BERT)

-Ready-to-use CLI, REST API, and optional simple web UI

🗂 Project Structure:
.
├── data/
│   ├── raw/                  # Original CSVs
│   ├── processed/            # Train/val/test splits
│   └── samples/              # Tiny sample CSVs for quick tests
├── models/
│   ├── checkpoints/          # Saved weights
│   └── tokenizer/            # Fitted tokenizer / vocab
├── src/
│   ├── config.py
│   ├── data.py               # Loading, cleaning, splitting
│   ├── model_bilstm.py       # BiLSTM + Attention
│   ├── model_bert.py         # BERT pipeline
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Metrics, curves
│   ├── infer.py              # CLI inference
│   └── utils.py
├── api/
│   ├── app.py                # FastAPI server
│   └── requirements.txt
├── ui/
│   └── app.py                # Streamlit/Gradio app
├── requirements.txt
├── README.md
└── LICENSE


🚀 Quickstart:

1) Clone & install:git clone https://github.com/KonukantiLaxman/my-major-project.git
cd my-major-project

2) Use sample data to sanity-check

3) Train on your dataset

-Place a CSV at data/raw/dataset.csv with columns:

-text: the post/comment/tweet

-label: 0 (non-bullying) or 1 (bullying) — can be extended to multi-class

📦 Datasets:

-This repo supports any CSV with text and label columns.

-Use src/data.py utilities to:

📊 Results:

<img width="1920" height="1080" alt="Screenshot 2025-05-03 095322" src="https://github.com/user-attachments/assets/0f0d709c-24d1-405e-9b0c-436bfe30e604" />

<img width="1920" height="1080" alt="Screenshot 2025-05-03 095348" src="https://github.com/user-attachments/assets/6febf033-8b54-45e8-bb82-b228dd112855" />

<img width="1920" height="1080" alt="Screenshot 2025-05-03 095410" src="https://github.com/user-attachments/assets/24fcee3a-0d77-4e09-addd-1bd11ac988bc" />

<img width="1920" height="1080" alt="Screenshot 2025-05-03 095603" src="https://github.com/user-attachments/assets/678d945f-9edd-4827-a232-cb9ee5583b15" />

<img width="1920" height="1080" alt="Screenshot 2025-05-03 100123" src="https://github.com/user-attachments/assets/85501b92-6c71-4e6e-bdc6-b0869027b016" />

<img width="1920" height="1080" alt="Screenshot 2025-05-04 210318" src="https://github.com/user-attachments/assets/e9b6eaf3-652f-4e4d-bca5-70b3c09de23a" />

📚 Cite This Work:

If this project helps your research or products, please cite:

@software{konukati2025cyberbullying,
  author  = {Konukati, Laxman},
  title   = {Deep Learning Neural Network for Cyberbullying Detection in Social Media},
  year    = {2025},
  url     = {https://github.com/KonukantiLaxman/my-major-project}


📩 Contact:

Laxman Konukati

Email: kl752008@gmail.com

LinkedIn: https://www.linkedin.com/in/konukati-laxman-6856092a5/

GitHub: https://github.com/KonukantiLaxman

📝 License:

This project is licensed under the MIT License — see LICENSE
 for details.






