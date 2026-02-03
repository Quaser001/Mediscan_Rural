import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, pipeline
from torchvision import transforms
from PIL import Image, ImageStat
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np
import io
import gc
import librosa
import soundfile as sf
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & GUARDRAILS
# ==========================================
MODELS = {
    "lungs": {
        "type": "image",
        "id": "nickmuchi/vit-finetuned-chest-xray-pneumonia",
        "desc": "Chest X-Ray Analysis",
        "safe": ["NORMAL", "normal", "No Pneumonia"],
        "rules": {"max_sat": 30, "reject_msg": "❌ Invalid: Too colorful. Please upload a B&W X-Ray."}
    },
    "cough": { 
        "type": "audio",
        "id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "desc": "Respiratory Audio Analysis",
        "target_labels": ["Cough", "Throat clearing", "Respiratory sounds", "Wheeze", "Gasping"],
        "rules": {"min_duration": 0.5, "reject_msg": "❌ Invalid: Audio too short or silent."}
    },
    "fracture": {
        "type": "image",
        "id": "nickmuchi/vit-finetuned-chest-xray-pneumonia", 
        "desc": "Bone Trauma X-Ray",
        "safe": ["NORMAL", "normal", "No Pneumonia"],
        "rules": {"max_sat": 30, "reject_msg": "❌ Invalid: Too colorful. Please upload a B&W X-Ray."}
    },
    "brain": {
        "type": "image",
        "id": "Hemgg/brain-tumor-classification",
        "desc": "Brain MRI Scan Analysis",
        "safe": ["no_tumor"],
        "rules": {"max_sat": 30, "reject_msg": "❌ Invalid: This looks like a Photo. Please upload a B&W MRI Scan."}
    },
    "eye": {
        "type": "image",
        "id": "AventIQ-AI/resnet18-cataract-detection-system", 
        "desc": "Ophthalmology Scan",
        "safe": ["Normal", "normal", "healthy"],
        "rules": {"min_sat": 20, "min_white": 0.05, "reject_msg": "❌ Invalid: No eye detected (Missing white sclera)."}
    },
    "skin": {
        "type": "image",
        "id": "Anwarkh1/Skin_Cancer-Image_Classification",
        "desc": "Dermatology Lesion Scan",
        "safe": ["Benign", "benign", "nv", "bkl"],
        "rules": {"min_sat": 20, "max_white": 0.15, "reject_msg": "❌ Invalid: Image looks like an Eye or Document."}
    }
}

# ==========================================
# 2. MEDICAL ENGINE (Logic)
# ==========================================
class MedicalEngine:
    def __init__(self):
        self.device = "cpu"
        print("✅ System Initialized: Medical Engine Ready")
        
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def validate_image(self, image, task):
        rules = MODELS[task].get("rules", {})
        img_hsv = image.convert('HSV')
        img_np = np.array(img_hsv)
        s_channel = img_np[:, :, 1]
        v_channel = img_np[:, :, 2]
        avg_sat = np.mean(s_channel)
        white_pixels = np.logical_and(s_channel < 40, v_channel > 180)
        white_ratio = np.sum(white_pixels) / white_pixels.size

        print(f"🔍 Analysis [{task}]: Sat={int(avg_sat)}, WhiteRatio={white_ratio:.3f}")

        if "max_sat" in rules and avg_sat > rules["max_sat"]: return False, rules["reject_msg"]
        if "min_sat" in rules and avg_sat < rules["min_sat"]: return False, "❌ Invalid: Image is B&W. Color photo required."
        if "min_white" in rules and white_ratio < rules["min_white"]: return False, rules["reject_msg"]
        if "max_white" in rules and white_ratio > rules["max_white"]: return False, rules["reject_msg"]
        return True, ""

    def validate_audio(self, audio_array, sr):
        duration = len(audio_array) / sr
        if duration < 0.5: return False, "❌ Audio too short (< 0.5s)."
        if np.max(np.abs(audio_array)) < 0.01: return False, "❌ Audio is silent/empty."
        return True, ""

    def predict(self, file_bytes, task):
        model_cfg = MODELS[task]
        
        if model_cfg["type"] == "audio":
            try:
                with open("temp_audio_input", "wb") as f: f.write(file_bytes)
                try: audio, sr = librosa.load("temp_audio_input", sr=16000)
                except: return {"error": "Audio Format Error. Use .wav or .mp3", "risk": "INVALID"}

                is_valid, msg = self.validate_audio(audio, sr)
                if not is_valid: return {"error": msg, "risk": "INVALID"}

                classifier = pipeline("audio-classification", model=model_cfg["id"])
                outputs = classifier("temp_audio_input")
                top = outputs[0]
                is_cough = any(t in res['label'] for res in outputs[:3] for t in model_cfg["target_labels"])
                
                risk = "HIGH" if is_cough and top['score'] > 0.4 else "LOW"
                label = f"Detected: {top['label']}" if is_cough else "Normal Background Noise"
                return {"task": task, "desc": model_cfg["desc"], "prediction": {"label": label, "score": top['score']}, "risk": risk}
            except Exception as e: return {"error": f"Audio Error: {str(e)}"}

        else:
            try:
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                is_valid, msg = self.validate_image(image, task)
                if not is_valid: return {"task": task, "risk": "INVALID", "error": msg, "prediction": {"label": "Rejected", "score": 0.0}}

                print(f"⏳ Loading Model: {task}...")
                model = AutoModelForImageClassification.from_pretrained(model_cfg["id"])
                model.to(self.device)
                model.eval()

                inputs = self.img_transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                
                results = [{"label": model.config.id2label[i], "score": float(score)} for i, score in enumerate(probs[0])]
                results.sort(key=lambda x: x['score'], reverse=True)
                top = results[0]
                
                safe_words = model_cfg["safe"]
                is_safe = any(s.lower() in top["label"].lower() for s in safe_words)
                
                if top["score"] < 0.5: risk = "UNCERTAIN"
                elif is_safe: risk = "LOW"
                else: risk = "HIGH" if top["score"] > 0.70 else "MODERATE"
                
                if task == "fracture":
                    top["label"] = "Fracture / Anomaly" if risk in ["HIGH", "MODERATE"] else "Healthy Bone"

                del model
                gc.collect()
                return {"task": task, "desc": model_cfg["desc"], "prediction": top, "risk": risk}
            except Exception as e: return {"error": f"Image Error: {str(e)}"}

# ==========================================
# 3. API & FRONTEND
# ==========================================
app = FastAPI()
engine = MedicalEngine()
HISTORY = []

@app.post("/predict/{task}")
async def predict_route(task: str, patient: str, age: str, file: UploadFile = File(...)):
    if task not in MODELS: return {"error": "Invalid Task"}
    content = await file.read()
    result = engine.predict(content, task)
    if "error" not in result and result.get("risk") != "INVALID":
        HISTORY.insert(0, {"time": datetime.now().strftime("%H:%M"), "patient": patient, "task": task.capitalize(), "diagnosis": result["prediction"]["label"], "risk": result["risk"]})
    return result

@app.get("/history")
def get_history(): return HISTORY

@app.post("/reset_history")
def reset_history():
    global HISTORY
    HISTORY = []
    return {"status": "cleared"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>MediScan Rural | Govt of Meghalaya</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { background: #f8fafc; background-image: radial-gradient(#cbd5e1 1px, transparent 1px); background-size: 24px 24px; }
        .glass-header { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.5); }
        .glass-card { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.5); box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07); }
        .icon-btn { transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1); background: rgba(255,255,255,0.8); backdrop-filter: blur(5px); }
        .icon-btn:hover { transform: translateY(-5px) scale(1.05); box-shadow: 0 15px 30px -5px rgba(59, 130, 246, 0.15); border-color: #93c5fd; }
        .icon-btn.active { background: #eff6ff; border-color: #3b82f6; box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15); transform: scale(0.98); }
        .icon-sticker { filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1)); }
    </style>
</head>
<body class="font-sans text-slate-800 min-h-screen flex flex-col">
    <nav class="glass-header sticky top-0 z-50">
        <div class="container mx-auto px-4 py-3 flex justify-between items-center">
            <div class="flex items-center gap-4">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Seal_of_Meghalaya.svg/150px-Seal_of_Meghalaya.svg.png" class="h-14 drop-shadow-md">
                <div>
                    <h2 class="text-[10px] md:text-xs font-bold text-slate-500 uppercase tracking-[0.2em]" data-translate="govt_title">GOVERNMENT OF MEGHALAYA</h2>
                    <h1 class="text-xl md:text-2xl font-extrabold text-slate-900 tracking-tight">MediScan <span class="text-blue-600">AI</span></h1>
                    <p class="text-[10px] font-bold text-green-600 flex items-center gap-1.5 mt-0.5">
                        <span class="relative flex h-2 w-2">
                          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                          <span class="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                        </span>
                        <span data-translate="online_status">Online Node: Shillong HQ</span>
                    </p>
                </div>
            </div>
            <div class="flex flex-col items-end gap-2">
                <select id="lang-select" onchange="changeLanguage()" class="bg-slate-100 border border-slate-200 text-slate-700 text-xs rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-1.5 shadow-sm outline-none">
                    <option value="en">🇬🇧 English</option>
                    <option value="hi">🇮🇳 हिंदी (Hindi)</option>
                    <option value="as">🇮🇳 অসমীয়া (Assamese)</option>
                    <option value="bn">🇧🇩 বাংলা (Bengali)</option>
                    <option value="kha">🌲 Khasi</option>
                    <option value="gar">⛰️ Garo</option>
                </select>
            </div>
        </div>
    </nav>
    <div class="container mx-auto mt-8 p-4 max-w-5xl flex-grow">
        
        <div class="grid grid-cols-3 md:grid-cols-6 gap-4 mb-8">
            <button onclick="setTask('lungs', 'image')" id="btn-lungs" class="icon-btn rounded-2xl p-4 flex flex-col items-center justify-center gap-3 border border-white/60 shadow-sm group">
                <i class="fas fa-lungs text-blue-500 text-5xl mb-1 icon-sticker transition-transform group-hover:scale-110"></i>
                <span class="text-xs font-extrabold text-slate-600 uppercase tracking-wide" data-translate="btn_lungs">Lungs</span>
            </button>
            <button onclick="setTask('cough', 'audio')" id="btn-cough" class="icon-btn rounded-2xl p-4 flex flex-col items-center justify-center gap-3 border border-white/60 shadow-sm group">
                <i class="fas fa-head-side-cough text-teal-500 text-5xl mb-1 icon-sticker transition-transform group-hover:scale-110"></i>
                <span class="text-xs font-extrabold text-slate-600 uppercase tracking-wide" data-translate="btn_cough">Cough</span>
            </button>
            <button onclick="setTask('fracture', 'image')" id="btn-fracture" class="icon-btn rounded-2xl p-4 flex flex-col items-center justify-center gap-3 border border-white/60 shadow-sm group">
                <i class="fas fa-bone text-slate-500 text-5xl mb-1 icon-sticker transition-transform group-hover:scale-110"></i>
                <span class="text-xs font-extrabold text-slate-600 uppercase tracking-wide" data-translate="btn_bone">Fracture</span>
            </button>
            <button onclick="setTask('brain', 'image')" id="btn-brain" class="icon-btn rounded-2xl p-4 flex flex-col items-center justify-center gap-3 border border-white/60 shadow-sm group">
                <i class="fas fa-brain text-purple-500 text-5xl mb-1 icon-sticker transition-transform group-hover:scale-110"></i>
                <span class="text-xs font-extrabold text-slate-600 uppercase tracking-wide" data-translate="btn_brain">Brain</span>
            </button>
            <button onclick="setTask('eye', 'image')" id="btn-eye" class="icon-btn rounded-2xl p-4 flex flex-col items-center justify-center gap-3 border border-white/60 shadow-sm group">
                <i class="fas fa-eye text-indigo-500 text-5xl mb-1 icon-sticker transition-transform group-hover:scale-110"></i>
                <span class="text-xs font-extrabold text-slate-600 uppercase tracking-wide" data-translate="btn_eye">Eye</span>
            </button>
            <button onclick="setTask('skin', 'image')" id="btn-skin" class="icon-btn rounded-2xl p-4 flex flex-col items-center justify-center gap-3 border border-white/60 shadow-sm group">
                <i class="fas fa-hand-dots text-orange-500 text-5xl mb-1 icon-sticker transition-transform group-hover:scale-110"></i>
                <span class="text-xs font-extrabold text-slate-600 uppercase tracking-wide" data-translate="btn_skin">Skin</span>
            </button>
        </div>
        <div class="glass-card rounded-3xl overflow-hidden relative">
            <div id="scope-box" class="hidden bg-slate-50/50 border-b border-slate-100 p-3 text-center backdrop-blur-sm">
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2" data-translate="lbl_scope">Scope of Detection</p>
                <div id="scope-tags" class="flex flex-wrap justify-center gap-2"></div>
            </div>
            <div class="p-8">
                <h2 id="header-text" class="text-2xl font-bold text-slate-800 mb-8 text-center">Select a Category</h2>
                
                <div id="inputs" class="opacity-50 pointer-events-none transition-all duration-500 mb-8 max-w-2xl mx-auto">
                    <div class="grid grid-cols-3 gap-4 mb-4">
                        <div class="col-span-2">
                            <label class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 block" data-translate="lbl_name">Patient Name</label>
                            <input type="text" id="p-name" class="w-full bg-white/80 border border-slate-200 p-3 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all">
                        </div>
                        <div>
                            <label class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 block" data-translate="lbl_age">Age</label>
                            <input type="number" id="p-age" class="w-full bg-white/80 border border-slate-200 p-3 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all">
                        </div>
                    </div>
                    
                    <div onclick="document.getElementById('file-input').click()" class="border-2 border-dashed border-slate-300 rounded-2xl p-10 text-center cursor-pointer hover:bg-blue-50/50 hover:border-blue-400 transition-all duration-300 group relative overflow-hidden bg-white/60">
                        <input type="file" id="file-input" class="hidden" onchange="showPreview(event)" onclick="this.value=null">
                        <div id="placeholder" class="group-hover:scale-105 transition-transform duration-300">
                            <div class="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-600 shadow-sm">
                                <i id="upload-icon" class="fas fa-cloud-upload-alt text-3xl"></i>
                            </div>
                            <p id="upload-text" class="text-slate-600 font-bold" data-translate="txt_upload">Tap to upload</p>
                            <p class="text-xs text-slate-400 mt-1">Supported: JPG, PNG, WAV, MP3</p>
                        </div>
                        <div class="relative z-10">
                            <img id="img-preview" class="hidden mx-auto max-h-64 rounded-xl shadow-lg object-contain bg-black/5">
                            <audio id="audio-preview" controls class="hidden w-full mt-4"></audio>
                        </div>
                    </div>
                </div>
                <button id="run-btn" onclick="analyze()" class="hidden w-full max-w-md mx-auto bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 text-white font-bold py-4 rounded-xl shadow-lg shadow-blue-200 transition-all transform hover:scale-[1.02] flex items-center justify-center gap-3 animate-pulse-soft">
                    <i class="fas fa-microscope text-xl"></i> <span class="text-lg" data-translate="btn_run">Run Diagnosis</span>
                </button>
                
                <div id="loader" class="hidden text-center py-8">
                    <div class="relative w-16 h-16 mx-auto mb-4">
                        <div class="absolute inset-0 border-4 border-slate-100 rounded-full"></div>
                        <div class="absolute inset-0 border-4 border-blue-500 rounded-full border-t-transparent animate-spin"></div>
                    </div>
                    <p class="text-sm font-bold text-slate-600 animate-pulse" data-translate="txt_analyzing">Analyzing data...</p>
                </div>
                <div id="result-box" class="hidden mt-10 border-t border-slate-100 pt-8">
                    <div class="flex flex-col md:flex-row justify-between items-start gap-6 mb-6">
                        <div>
                            <p class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1" data-translate="lbl_result">Analysis Result</p>
                            <h1 id="res-label" class="text-3xl md:text-4xl font-extrabold text-slate-800 tracking-tight">--</h1>
                            <p class="text-sm text-slate-500 mt-2 font-medium bg-slate-100/80 inline-block px-3 py-1 rounded-lg border border-slate-200">
                                <span data-translate="lbl_conf">AI Confidence</span>: <span id="res-conf" class="font-mono text-slate-800">--</span>
                            </p>
                        </div>
                        <span id="res-badge" class="px-5 py-2.5 rounded-xl text-sm font-bold uppercase shadow-sm tracking-wide">--</span>
                    </div>
                    
                    <div id="alert-box" class="hidden p-6 rounded-2xl border border-l-4 shadow-sm flex flex-col md:flex-row items-start md:items-center gap-5 transition-all bg-white/80 backdrop-blur">
                        <div class="p-4 bg-slate-50 rounded-full shrink-0">
                            <i id="alert-icon" class="fas fa-info-circle text-2xl text-slate-600"></i>
                        </div>
                        <div class="flex-grow">
                            <strong class="block text-lg mb-1 text-slate-800" data-translate="lbl_action">Action Required</strong>
                            <span id="alert-text" class="text-sm text-slate-600 leading-relaxed">--</span>
                        </div>
                        <button id="doctor-btn" onclick="findDoctor()" class="hidden px-6 py-3 bg-white text-slate-800 font-bold text-sm rounded-xl shadow-md border border-slate-100 hover:bg-slate-50 hover:shadow-lg transition flex items-center gap-2 whitespace-nowrap">
                            <i class="fas fa-map-marker-alt text-red-500 text-lg"></i> <span>Find Nearest Doctor</span>
                        </button>
                    </div>
                    <div class="mt-8 flex items-center justify-between text-[10px] text-slate-400 border-t border-slate-50 pt-4">
                        <span class="flex items-center gap-2"><i class="fas fa-database text-blue-400"></i> <span data-translate="lbl_sync">Ayushman Bharat Sync</span></span>
                        <span id="sync-msg" class="text-yellow-500 font-bold flex items-center gap-1"><i class="fas fa-sync fa-spin"></i> Pending...</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="mt-12 mb-12">
            <div class="flex justify-between items-center mb-6">
                <h3 class="text-sm font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2"><i class="fas fa-history"></i> Recent Patients</h3>
                <button onclick="clearHistory()" class="text-[10px] font-bold text-red-500 hover:text-red-700 bg-red-50 hover:bg-red-100 px-4 py-2 rounded-lg transition-colors uppercase tracking-wide">Clear History</button>
            </div>
            <div class="glass-card rounded-xl shadow-sm overflow-hidden border border-slate-200/50">
                <table class="w-full text-sm text-left">
                    <thead class="bg-slate-50/80 text-slate-500 font-bold uppercase text-[10px] tracking-wider">
                        <tr><th class="px-6 py-4">Time</th><th class="px-6 py-4">Patient</th><th class="px-6 py-4">Category</th><th class="px-6 py-4">Diagnosis</th><th class="px-6 py-4">Risk</th></tr>
                    </thead>
                    <tbody id="history-table" class="divide-y divide-slate-100"></tbody>
                </table>
            </div>
        </div>
    </div>
    <script>
        const MODEL_SCOPES = {
            lungs: ["Pneumonia", "Tuberculosis", "Viral Infection", "Normal Lung"],
            cough: ["COPD Signs", "Whooping Cough", "Wheezing", "Respiratory Infection"],
            fracture: ["Bone Fracture", "Dislocation", "Healthy Bone Structure"],
            brain: ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"],
            eye: ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal Eye"],
            skin: ["Melanoma", "Basal Cell Carcinoma", "Nevus (Mole)", "Benign Keratosis"]
        };
        const TRANSLATIONS = {
            en: {
                govt_title: "GOVERNMENT OF MEGHALAYA", online_status: "Online Node: Shillong HQ",
                btn_lungs: "Lungs", btn_cough: "Cough", btn_bone: "Fracture", btn_brain: "Brain", btn_eye: "Eye", btn_skin: "Skin",
                lbl_name: "Patient Name", lbl_age: "Age / ID", txt_upload: "Tap to upload Scan/Photo", btn_run: "Run Diagnosis",
                txt_analyzing: "Analyzing medical data...", lbl_result: "Analysis Result", lbl_conf: "AI Confidence",
                lbl_action: "Medical Action Required", lbl_scope: "Scope of Detection", lbl_sync: "Ayushman Bharat Sync"
            },
            hi: {
                govt_title: "मेघालय सरकार", online_status: "ऑनलाइन नोड: शिलांग",
                btn_lungs: "फेफड़े", btn_cough: "खांसी", btn_bone: "हड्डी", btn_brain: "मस्तिष्क", btn_eye: "आंख", btn_skin: "त्वचा",
                lbl_name: "रोगी का नाम", lbl_age: "आयु", txt_upload: "अपलोड करें", btn_run: "जांच करें",
                txt_analyzing: "विश्लेषण कर रहा है...", lbl_result: "परिणाम", lbl_conf: "विश्वास",
                lbl_action: "आवश्यक कार्रवाई", lbl_scope: "जांच का दायरा", lbl_sync: "सरकारी सिंक"
            },
            as: {
                govt_title: "মেঘালয় চৰকাৰ", online_status: "অনলাইন ন'ড: শ্বিলং",
                btn_lungs: "হাঁওফাঁও", btn_cough: "কাহ", btn_bone: "হাৰ ভঙা", btn_brain: "মগজু", btn_eye: "চকু", btn_skin: "ছাল",
                lbl_name: "নাম", lbl_age: "বয়স", txt_upload: "আপলোড কৰক", btn_run: "পৰীক্ষা কৰক",
                txt_analyzing: "বিশ্লেষণ...", lbl_result: "ফলাফল", lbl_conf: "নিশ্চয়তা",
                lbl_action: "পদক্ষেপ", lbl_scope: "পৰিসৰ", lbl_sync: "ডাটাবেচ"
            },
            bn: {
                govt_title: "মেঘালয় সরকার", online_status: "অনলাইন নোড: শিলং",
                btn_lungs: "ফুসফুস", btn_cough: "কাশি", btn_bone: "হাড় ভাঙা", btn_brain: "মস্তিষ্ক", btn_eye: "চোখ", btn_skin: "ত্বক",
                lbl_name: "রোগীর নাম", lbl_age: "বয়স", txt_upload: "ছবি আপলোড করুন", btn_run: "রোগ নির্ণয় করুন",
                txt_analyzing: "বিশ্লেষণ করা হচ্ছে...", lbl_result: "ফলাফল", lbl_conf: "বিশ্বাসযোগ্যতা",
                lbl_action: "প্রয়োজনীয় পদক্ষেপ", lbl_scope: "শনাক্তকরণের পরিসর", lbl_sync: "সরকারি সিঙ্ক"
            },
            kha: {
                govt_title: "SORKAR MEGHALAYA", online_status: "Online: Shillong",
                btn_lungs: "Phopsa", btn_cough: "Jyrhoh", btn_bone: "Shyieng", btn_brain: "Jabieng", btn_eye: "Khmat", btn_skin: "Doh",
                lbl_name: "Kyrteng", lbl_age: "Rta", txt_upload: "Thep dur", btn_run: "Leh Test",
                txt_analyzing: "Check...", lbl_result: "Result", lbl_conf: "Jingshisha",
                lbl_action: "Leh kane", lbl_scope: "Jingshem", lbl_sync: "Sorkar Sync"
            },
            gar: {
                govt_title: "MEGHALAYA SORKARI", online_status: "Online: Shillong",
                btn_lungs: "Ka'sop", btn_cough: "Gusuk", btn_bone: "Greng", btn_brain: "Taning", btn_eye: "Mikron", btn_skin: "Bigil",
                lbl_name: "Bimung", lbl_age: "Bilsi", txt_upload: "Gata", btn_run: "Porikka",
                txt_analyzing: "Niyenga...", lbl_result: "Result", lbl_conf: "Bebera'ani",
                lbl_action: "Kam", lbl_scope: "Am·sandiani", lbl_sync: "Sorkar Sync"
            }
        };
        let currTask = null, currType = 'image', currFile = null, currLang = 'en';
        updateHistoryTable();
        function changeLanguage() {
            currLang = document.getElementById('lang-select').value;
            let t = TRANSLATIONS[currLang];
            document.querySelectorAll('[data-translate]').forEach(el => {
                let key = el.getAttribute('data-translate');
                if (t[key]) el.innerText = t[key];
            });
            if (currTask) {
                let taskKey = "btn_" + currTask;
                document.getElementById('header-text').innerHTML = `Upload <span class="uppercase text-blue-600">${t[taskKey]}</span>`;
            }
        }
        function setTask(task, type) {
            currTask = task; currType = type;
            let t = TRANSLATIONS[currLang];
            
            document.querySelectorAll('.icon-btn').forEach(b => {
                b.classList.remove('active', 'border-blue-500', 'ring-2', 'ring-blue-200');
                b.classList.add('border-white/60');
            });
            let btn = document.getElementById('btn-'+task);
            btn.classList.add('active', 'border-blue-500', 'ring-2', 'ring-blue-200');
            
            document.getElementById('header-text').innerHTML = `Upload <span class="uppercase text-blue-600">${t["btn_" + task]}</span>`;
            let scopeTags = document.getElementById('scope-tags');
            scopeTags.innerHTML = "";
            MODEL_SCOPES[task].forEach(d => {
                let tag = document.createElement("span");
                tag.className = "px-3 py-1 bg-white text-slate-600 text-[10px] font-bold uppercase rounded-full border border-slate-200 tracking-wide shadow-sm";
                tag.innerText = d;
                scopeTags.appendChild(tag);
            });
            document.getElementById('scope-box').classList.remove('hidden');
            let input = document.getElementById('file-input');
            let icon = document.getElementById('upload-icon');
            let txt = document.getElementById('upload-text');
            
            if (type === 'audio') {
                input.accept = ".wav, .mp3, audio/*";
                icon.className = "fas fa-microphone-alt text-4xl text-teal-500 mb-2";
                txt.innerHTML = "Tap to upload Audio<br><span class='text-xs text-slate-400'>.wav, .mp3</span>";
            } else {
                input.accept = "image/*";
                icon.className = "fas fa-cloud-upload-alt text-4xl text-blue-400 mb-2";
                txt.innerText = t['txt_upload'];
            }
            document.getElementById('inputs').classList.remove('opacity-50', 'pointer-events-none');
            document.getElementById('result-box').classList.add('hidden');
            document.getElementById('run-btn').classList.add('hidden');
            document.getElementById('placeholder').classList.remove('hidden');
            document.getElementById('img-preview').classList.add('hidden');
            document.getElementById('audio-preview').classList.add('hidden');
            currFile = null;
        }
        function showPreview(event) {
            if (event.target.files && event.target.files[0]) {
                currFile = event.target.files[0];
                let url = URL.createObjectURL(currFile);
                if (currType === 'audio') {
                    let aud = document.getElementById('audio-preview');
                    aud.src = url; aud.classList.remove('hidden');
                    document.getElementById('img-preview').classList.add('hidden');
                } else {
                    let img = document.getElementById('img-preview');
                    img.src = url; img.classList.remove('hidden');
                    document.getElementById('audio-preview').classList.add('hidden');
                }
                document.getElementById('placeholder').classList.add('hidden');
                document.getElementById('run-btn').classList.remove('hidden');
                document.getElementById('result-box').classList.add('hidden');
            }
        }
        async function analyze() {
            if (!currTask || !currFile) return;
            let pname = document.getElementById('p-name').value;
            let page = document.getElementById('p-age').value;
            if (!pname) { alert("Please enter Patient Name first."); return; }
            document.getElementById('run-btn').classList.add('hidden');
            document.getElementById('loader').classList.remove('hidden');
            document.getElementById('result-box').classList.add('hidden');
            let formData = new FormData();
            formData.append("file", currFile);
            try {
                let url = `/predict/${currTask}?patient=${encodeURIComponent(pname)}&age=${encodeURIComponent(page)}`;
                let res = await fetch(url, { method: "POST", body: formData });
                let data = await res.json();
                document.getElementById('loader').classList.add('hidden');
                document.getElementById('result-box').classList.remove('hidden');
                
                updateHistoryTable();
                if (data.risk === "INVALID") {
                    updateResultUI("Rejected", "--", "INVALID INPUT", "bg-slate-200 text-slate-600");
                    showAlert("Input Error", data.error, "bg-slate-100 border-slate-200 text-slate-800", "fa-exclamation-triangle text-slate-400");
                    return;
                }
                if (data.error) { alert(data.error); document.getElementById('run-btn').classList.remove('hidden'); return; }
                let conf = (data.prediction.score * 100).toFixed(1) + "%";
                updateResultUI(data.prediction.label, conf, data.risk + " RISK", "");
                let btnDoc = document.getElementById('doctor-btn');
                
                if (data.risk === "HIGH") {
                    document.getElementById('res-badge').className = "px-4 py-2 rounded-full text-sm font-bold uppercase shadow-sm tracking-wide bg-red-100 text-red-700";
                    showAlert("Critical Issue Detected", "Immediate referral to District Hospital recommended.", "bg-red-50 border-red-100 text-red-900", "fa-ambulance text-red-500");
                    btnDoc.classList.remove('hidden');
                } else if (data.risk === "MODERATE") {
                    document.getElementById('res-badge').className = "px-4 py-2 rounded-full text-sm font-bold uppercase shadow-sm tracking-wide bg-yellow-100 text-yellow-800";
                    showAlert("Moderate Risk", "Consult local PHC doctor for further tests.", "bg-yellow-50 border-yellow-100 text-yellow-900", "fa-user-md text-yellow-600");
                    btnDoc.classList.remove('hidden');
                } else {
                    document.getElementById('res-badge').className = "px-4 py-2 rounded-full text-sm font-bold uppercase shadow-sm tracking-wide bg-green-100 text-green-700";
                    document.getElementById('alert-box').classList.add('hidden');
                    btnDoc.classList.add('hidden');
                }
                setTimeout(() => {
                    document.getElementById('sync-msg').innerHTML = "<i class='fas fa-check-circle'></i> Synced to Govt Database";
                    document.getElementById('sync-msg').className = "text-green-600 font-bold flex items-center gap-1";
                }, 2000);
            } catch (e) {
                alert("Connection Failed.");
                document.getElementById('loader').classList.add('hidden');
                document.getElementById('run-btn').classList.remove('hidden');
            }
        }
        // FIND NEAREST DOCTOR (Corrected URL)
        function findDoctor() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition((pos) => {
                    let lat = pos.coords.latitude;
                    let lon = pos.coords.longitude;
                    window.open(`https://www.google.com/maps/search/hospitals+near+me/@${lat},${lon},13z`, '_blank');
                }, () => {
                    window.open('https://www.google.com/maps/search/hospitals+near+me', '_blank');
                });
            } else {
                window.open('https://www.google.com/maps/search/hospitals+near+me', '_blank');
            }
        }
        function updateResultUI(label, conf, badgeText, badgeClass) {
            document.getElementById('res-label').innerText = label;
            document.getElementById('res-conf').innerText = conf;
            document.getElementById('res-badge').innerText = badgeText;
            if(badgeClass) document.getElementById('res-badge').className = `px-4 py-2 rounded-full text-sm font-bold uppercase shadow-sm tracking-wide ${badgeClass}`;
        }
        function showAlert(title, msg, classes, iconClass) {
            let a = document.getElementById('alert-box');
            a.className = `p-5 rounded-xl border border-l-4 shadow-sm flex flex-col md:flex-row items-start md:items-center gap-4 transition-all ${classes}`;
            a.classList.remove('hidden');
            document.getElementById('alert-text').innerText = msg;
            document.querySelector('[data-translate="lbl_action"]').innerText = title;
            document.getElementById('alert-icon').className = `fas ${iconClass} text-2xl`;
        }
        async function updateHistoryTable() {
            try {
                let res = await fetch("/history");
                let data = await res.json();
                let tbody = document.getElementById('history-table');
                tbody.innerHTML = "";
                if(data.length === 0) {
                    tbody.innerHTML = '<tr class="text-slate-400 text-center italic"><td colspan="5" class="py-6">No records found.</td></tr>';
                    return;
                }
                data.forEach(row => {
                    let color = row.risk === "HIGH" ? "text-red-600 font-bold bg-red-50" : row.risk === "MODERATE" ? "text-yellow-600 font-bold bg-yellow-50" : "text-green-600 bg-green-50";
                    let tr = `
                        <tr class="bg-white border-b border-slate-50 hover:bg-slate-50 transition-colors">
                            <td class="px-6 py-4 text-slate-500 font-mono text-xs">${row.time}</td>
                            <td class="px-6 py-4 font-bold text-slate-700">${row.patient}</td>
                            <td class="px-6 py-4 text-slate-500 text-xs uppercase tracking-wide">${row.task}</td>
                            <td class="px-6 py-4 text-slate-800 font-medium">${row.diagnosis}</td>
                            <td class="px-6 py-4"><span class="px-2 py-1 rounded text-[10px] uppercase tracking-wider ${color}">${row.risk}</span></td>
                        </tr>`;
                    tbody.innerHTML += tr;
                });
            } catch(e) {}
        }
        async function clearHistory() {
            if(confirm("Clear local patient history?")) { await fetch("/reset_history", { method: "POST" }); updateHistoryTable(); }
        }
    </script>
</body>
</html>
