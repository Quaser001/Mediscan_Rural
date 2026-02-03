# MediScan Rural AI 🏥
**Government of Meghalaya** | *Bridging the Healthcare Gap with AI*

![MediScan Status](https://img.shields.io/badge/Status-Live-green) ![Platform](https://img.shields.io/badge/Platform-FastAPI-blue) ![AI](https://img.shields.io/badge/AI-PyTorch%20%7C%20Transformers-orange)

MediScan Rural is an advanced AI-powered diagnostic support system designed for rural healthcare centers with limited connectivity. It enables frontline health workers (ASHAs, ANMs) to perform screening for critical conditions using simple inputs like X-rays, photos, or audio recordings.

## 🌟 Key Features

### 🧠 Multi-Modal AI Diagnosis
The system integrates state-of-the-art models for 6 key medical tasks:
- **🫁 Lungs**: Detects Pneumonia from Chest X-Rays (`VIT-Chest-Xray`).
- **🤧 Cough**: Analyzes audio for respiratory diseases like COPD and Whooping Cough (`AST-Audio-Spectrogram`).
- **🦴 Fracture**: Identifies bone fractures and anomalies from X-Rays.
- **🧠 Brain**: Classifies MRI scans for presence of tumors (`Glioma`, `Meningioma`, `Pituitary`).
- **👁️ Eye**: Screens for Cataracts, Diabetic Retinopathy, and Glaucoma.
- **✋ Skin**: Analyzes lesions for Melanoma and other dermatological conditions.

### 🛡️ Smart Guardrails & Quality Control
Prevents garbage-in-garbage-out with real-time validation:
- **Image Validation**: Checks for correct modality (e.g., rejects color photos for X-ray tasks).
- **Audio Validation**: Ensures minimum duration and non-silent audio for cough analysis.
- **Rejection Logic**: Automatically flags invalid inputs with clear, localized error messages.

### 🌏 Hyper-Localized & Accessibility
Built for the diverse linguistic landscape of Meghalaya and North East India:
- **Multilingual UI**: Instant switching between **English, Hindi, Assamese, Bengali, Khasi, and Garo**.
- **Visual-First Design**: Icon-heavy interface for ease of use by workers with varying literacy levels.
- **Ayushman Bharat Sync**: Placeholder integration for syncing patient records with national health databases.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Quaser001/Mediscan_Rural.git
   cd Mediscan_Rural
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # Note: For audio processing, you may need ffmpeg installed on your system.
   ```

3. **Run the Application**
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the Interface**
   Open your browser and navigate to: `http://127.0.0.1:8000`

---

## 📋 API Usage

The backend exposes a REST API for integration with other systems.

**Endpoint**: `POST /predict/{task}`

**Parameters**:
- `task`: `lungs` | `cough` | `fracture` | `brain` | `eye` | `skin`
- `patient`: Patient Name (String)
- `age`: Patient Age (String/Int)
- `file`: Binary file upload (Image or Audio)

**Response Example**:
```json
{
  "task": "lungs",
  "desc": "Chest X-Ray Analysis",
  "prediction": {
    "label": "Pneumonia",
    "score": 0.98
  },
  "risk": "HIGH"
}
```

---

## 🛠️ Technology Stack
- **Frontend**: HTML5, TailwindCSS (CDN), Vanilla JS
- **Backend**: FastAPI (Python)
- **AI/ML Engine**: PyTorch, Hugging Face Transformers
- **Audio Processing**: Librosa, SoundFile

---

## ⚠️ Disclaimer
> **MediScan Rural AI is a Clinical Decision Support System (CDSS).**
> It is NOT a replacement for a qualified medical professional. All "High Risk" predictions must be verified by a doctor or specialist. The system is intended to triage cases and prioritize critical patients in resource-constrained settings.
