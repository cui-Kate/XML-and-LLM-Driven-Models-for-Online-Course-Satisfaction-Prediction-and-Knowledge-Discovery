# 1. Dataset

File: data.csv

Description: Original dataset used for quantitative modeling.

Source: Publicly available at Kaggle – Online Education System Review




# 2. Code Files

**model.ipynb**

Core notebook for explainable machine learning analysis.

Outputs include:

Descriptive statistics of the dataset

Trained model weights (.pkl files)

Model evaluation results with visualizations

SHAP-based interpretation with global/local plots

**LDA.py**

Script for topic modeling of interview transcripts.

Produces latent topics using LDA (Latent Dirichlet Allocation).



**CAM.py**


Script for extracting cross-contextual mechanisms.

Outputs:

mechanism_clusters.json — clustered mechanism structures

analysis_report.txt — textual explanation and summary



# 3. Supporting Materials (Appendices)

Interview outline.docx

Semi-structured interview guide, corresponding to Appendix A of the manuscript.

LLM prompts for text analysis.docx

Structured prompts designed for qualitative text analysis using LLMs, corresponding to Appendix B.

encode_result.txt

Encoded outputs from LLM (GPT-5 specified), applied to interview transcripts.

Serves as the empirical result file for Appendix C.


# 4. Project Overview

This repository implements a mixed-methods research design integrating:

Quantitative modeling: explainable machine learning (Random Forest, XGBoost, etc.) for predicting online course satisfaction.

Qualitative analysis: LLM-as-Encoder, LDA topic modeling, and mechanism extraction for theory-building.

Knowledge integration: alignment of SHAP explanations with qualitative mechanisms to generate evidence-based causal reasoning narratives.

The repository provides both reproducible code and supporting documentation for academic replication and methodological demonstration.



# 5. How to Run
**Step 1: Environment Setup**


Recommended environment: Python 3.10+

**Step 2: Quantitative Modeling**

Open and execute model.ipynb in Jupyter Notebook or VS Code.

This will:

Load and preprocess data.csv

Train multiple machine learning models

Save trained weights (.pkl files)

Generate evaluation results and visualizations

Produce SHAP-based interpretability plots

**Step 3: Qualitative Topic Modeling**

Run the LDA script:  python LDA.py

**Step 4: Mechanism Extraction**

Run the CAM script:  python CAM.py
