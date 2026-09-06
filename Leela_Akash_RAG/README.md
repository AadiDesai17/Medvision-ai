Objective

Retrieve relevant medical knowledge for the disease or condition predicted by the medical imaging model and provide that context to the patient-friendly explanation module.

RAG Pipeline
Medical Knowledge Documents
        ↓
    Text Chunking
        ↓
Sentence Transformer Embeddings
        ↓
     FAISS Index
        ↓
Predicted Condition Query
        ↓
Retrieve Relevant Passages
        ↓
    Context for LLM
        ↓
Patient-Friendly Explanation
Dataset Groups
1. Chest X-ray — ChestX-ray14

14 findings:

Atelectasis
Cardiomegaly
Effusion
Infiltration
Mass
Nodule
Pneumonia
Pneumothorax
Consolidation
Edema
Emphysema
Fibrosis
Pleural Thickening
Hernia
2. Skin Lesions — HAM10000

7 classes:

Actinic Keratosis
Basal Cell Carcinoma
Benign Keratosis
Dermatofibroma
Melanoma
Melanocytic Nevus
Vascular Lesion
3. Brain Tumor MRI

4 classes:

Glioma
Meningioma
Pituitary Tumor
No Tumor
4. Diabetic Retinopathy — EyePACS

5 grades:

No Diabetic Retinopathy
Mild Diabetic Retinopathy
Moderate Diabetic Retinopathy
Severe Diabetic Retinopathy
Proliferative Diabetic Retinopathy
Folder Structure
Leela_Akash_RAG/
│
├── medical_knowledge/
│   ├── chest_xray/
│   ├── skin/
│   ├── brain/
│   └── eye/
│
├── rag/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── ingest.py
│   └── retrieve.py
│
├── data/
│   └── faiss/
│
├── test_rag.py
├── requirements.txt
└── README.md
Technologies
Python
Sentence Transformers
all-MiniLM-L6-v2
FAISS
NumPy
How It Works

The medical knowledge documents are divided into smaller chunks. Each chunk is converted into a numerical embedding using a Sentence Transformer.

The embeddings are stored in a FAISS vector index.

When the disease prediction module produces a condition such as Pneumonia, that condition is converted into a query embedding. FAISS searches the index and returns the most relevant medical passages.

The retrieved passages are then passed to the LLM module for generating a patient-friendly explanation.

Installation
pip install -r requirements.txt
Run the RAG Pipeline

Create the FAISS index:

python -m rag.ingest

Test retrieval:

python test_rag.py
Example
prediction = "Pneumonia"

results = retriever.retrieve(
    prediction,
    top_k=5
)

The retrieved medical context can then be passed to the explanation module.

Safety Note

This RAG module is intended for the MedVision academic prototype and provides general medical information. It must not be treated as a medical diagnosis or as a substitute for advice from a qualified healthcare professional.
