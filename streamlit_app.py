import os
import re
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Page config ───────────────────────────────────────────
_icon = Image.open(os.path.join(BASE_DIR, "static", "icon.png"))
st.set_page_config(
    page_title="Intelligent Systems — AI Web App",
    page_icon=_icon,
    layout="wide",
)

st.markdown("""
<style>
.stButton>button{width:100%}
</style>
""", unsafe_allow_html=True)

# ─── Load models (cached) ───────────────────────────────────
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_models():
    bike = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model", "bigbike_high_precision_v2.keras")
    )
    gamble = joblib.load(
        os.path.join(BASE_DIR, "model", "ensemble_gambling_model_v2.joblib")
    )
    return bike, gamble

bike_model, gambling_model = load_models()
BIKE_CLASS_NAMES = ["big_bike_500cc", "small_bike"]

# ─── Helper functions ───────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', text)
    return text.strip()

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ─── UI ────────────────────────────────────────────────────
st.title("Intelligent Systems — AI Web App")
st.caption("ระบบ AI 2 โมเดล: ตรวจจำแนกรถจักรยานยนต์ & ตรวจจับข้อความพนัน")

tab1, tab2 = st.tabs(["Motorcycle Classifier", "Gambling Text Detector"])

# ══════════════════════════════════════════════════════════
#  Tab 1 — Motorcycle Classifier
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('[ทดสอบโมเดลนี้ คลิกที่นี่ (เลื่อนไปส่วน "ใช้งานโมเดล")](#motorcycle-use-model)')
    st.subheader("ข้อมูลโมเดล")
    st.markdown("""
 🏍️ Project: Motorcycle 500cc Classification AI

โปรเจกต์นี้พัฒนาขึ้นเพื่อจำแนกรถจักรยานยนต์ผ่านรูปภาพ โดยเน้นการแยกแยะระหว่างรถบิ๊กไบค์ (500cc ขึ้นไป) และรถรุ่นเล็ก เพื่อใช้ในการบริหารจัดการพื้นที่จอดรถหรือการคัดกรองยานพาหนะอัตโนมัติ

---

###  1. การเตรียมข้อมูล (Data Preparation)
หัวใจสำคัญของโปรเจกต์คือชุดข้อมูลที่มีคุณภาพสูงและสะท้อนความเป็นจริง:

* **Dataset Structure**: จัดกลุ่มรถออกเป็น 2 คลาส ได้แก่ 
    * `Allowed`: กลุ่มรถที่อนุญาต (รุ่น 500cc ขึ้นไป) 
    * `Not Allowed`: กลุ่มรถที่ไม่อนุญาต (รุ่นต่ำกว่า 500cc)
* **Image Sourcing**: ใช้เทคนิค Automated Scraping ดึงรูปภาพจาก Search Engine โดยเน้น Keyword เฉพาะเจาะจง เช่น "model name + motorcycle side view" เพื่อให้ AI เห็นองค์ประกอบของตัวรถจากด้านข้างอย่างชัดเจน
* **Data Sanitization**: ตรวจสอบไฟล์ภาพด้วยไลบรารี PIL เพื่อลบไฟล์ที่เสียหายหรือดาวน์โหลดไม่สมบูรณ์ (Corrupted Data)
* **Data Augmentation**: ใช้เทคนิคการจำลองภาพ (Rotation, Zoom, Horizontal Flip) เพื่อให้โมเดลมีความทนทาน (Robustness) ต่อมุมกล้องและสภาพแสง

---

###  2. ความท้าทาย: ปัญหาความคล้ายคลึงของรูปลักษณ์ (Visual Ambiguity)
ความท้าทายหลักของโปรเจกต์นี้คือ **Inter-class Similarity** หรือการที่รถคนละคลาสมีดีไซน์ที่ใกล้เคียงกันมาก โดยเฉพาะรถทรง **Sport Full Fairing**

###  กรณีศึกษา: รถทรงสปอร์ต (Sport Bikes)
แม้จะมีรูปลักษณ์ภายนอกที่ดู "โฉบเฉี่ยว" เหมือนกัน แต่โมเดลต้องเรียนรู้จุดแตกต่างเชิงลึก (Fine-grained Features) ดังนี้:

| ลักษณะเปรียบเทียบ | **Honda CBR1000RR** (Allowed - 1000cc) | **Yamaha R15** (Not Allowed - 155cc) |
| :--- | :--- | :--- |
| **ระบบเบรกคู่หน้า** | ดิสก์เบรกคู่ (Double Disc Brake) ขนาดใหญ่ | ดิสก์เบรกเดี่ยว (Single Disc Brake) |
| **ขนาดยางและล้อ** | ยางหลังกว้าง (180-190mm) สวิงอาร์มหนา | ยางแคบกว่า สวิงอาร์มมีขนาดเล็ก |
| **ห้องเครื่อง** | เครื่องยนต์เต็มเฟรม ท่อไอเสียและคอท่อขนาดใหญ่ | มีช่องว่างในเฟรมมากกว่า ท่อไอเสียขนาดเล็ก |
| **มิติตัวรถ** | ตัวถังน้ำมันกว้างและดูบึกบึน (Bulkiness) | ตัวรถเพรียวบางและน้ำหนักเบากว่า |

---

###  3. ทฤษฎีของอัลกอริทึม (Algorithm Theory)
เลือกใช้สถาปัตยกรรม **MobileNetV2** ซึ่งเป็น Deep Learning Model ที่มีประสิทธิภาพสูงและประหยัดทรัพยากร:

* **Convolutional Neural Network (CNN)**: จำลองการทำงานของประสาทตาในการตรวจจับเส้นขอบ พื้นผิว และรูปทรง
* **Transfer Learning**: นำโมเดลที่ผ่านการเทรนจากชุดข้อมูล ImageNet มาปรับใช้ เพื่อให้ AI มีความรู้พื้นฐานเรื่องรูปทรงวัตถุตั้งแต่วันแรก
* **Fine-Tuning Strategy**: การปลดล็อกเลเยอร์ส่วนท้าย (Unfreeze) เพื่อปรับจูนน้ำหนักให้เข้ากับลักษณะเฉพาะของมอเตอร์ไซค์โดยเฉพาะ

---

###  4. ขั้นตอนการพัฒนาโมเดล (Model Development)
1.  **Architecture Setup**: วางโครงสร้าง Sequential Model ประกอบด้วย MobileNetV2 ตามด้วย Global Average Pooling และ Dense Layer
2.  **Regularization**: เพิ่ม Batch Normalization และ Dropout Rate (0.4) เพื่อป้องกันอาการ Overfitting
3.  **Hyperparameter Tuning**: กำหนด Learning Rate ไว้ที่ระดับต่ำ (1e-5) เพื่อความละเอียดในการปรับจูน
4.  **Training Callbacks**: 
    * `EarlyStopping`: หยุดการเทรนเมื่อความแม่นยำเริ่มคงที่
    * `ReduceLROnPlateau`: ปรับลดความเร็วการเรียนรู้เมื่อเข้าใกล้จุดที่แม่นยำที่สุด

---

###  5. สถานะปัจจุบันและการวิเคราะห์ผลลัพธ์ (Current Status & Analysis)
ในเฟสเริ่มต้น โมเดลสามารถแสดงผลลัพธ์การจำแนกเบื้องต้นได้ (Proof of Concept) อย่างไรก็ตาม ยังพบข้อผิดพลาดในบางกรณี (Misclassification) ซึ่งมีสาเหตุทางเทคนิคดังนี้:

* **Data Scarcity**: เนื่องจากชุดข้อมูลปัจจุบันยังมีจำนวนจำกัด ทำให้โมเดลยังไม่สามารถสร้างการเรียนรู้แบบ Generalization ที่สมบูรณ์ได้ในทุกสภาพแสงหรือทุกมุมมอง
* **Feature Bias**: ในบางรูปโมเดลอาจตัดสินจากองค์ประกอบรอง (เช่น สีของรถ หรือ ฉากหลัง) มากกว่าจุดตัดสินหลัก (Key Features) เนื่องจากปริมาณข้อมูลที่ใช้สอนยังไม่ครอบคลุม "ความยาก" ของเคสรถทรงสปอร์ตที่หน้าตาคล้ายกัน
* **Performance Insight**: แม้จะมีการ "เดา" ผิดในบางครั้ง แต่ทิศทางการเรียนรู้เริ่มแสดงให้เห็นว่าแนวทาง Transfer Learning สามารถระบุคุณลักษณะพื้นฐานของยานพาหนะได้ถูกต้อง (Baseline Performance) และพร้อมที่จะพัฒนาต่อด้วยชุดข้อมูลที่เข้มข้นขึ้น

---
###  6. แหล่งอ้างอิงข้อมูล (References)
* **Model Architecture**: MobileNetV2: Inverted Residuals and Linear Bottlenecks (Google Research)
* **Library**: TensorFlow & Keras Framework (Google OSS)
* **Data Provider**: Bing Image Search API via `bing-image-downloader`
* **Implementation Guide**: Keras API Documentation (https://keras.io/)
""")

    st.divider()

    st.subheader("ใช้งานโมเดลและผลลัพธ์", anchor="motorcycle-use-model")

    if "bike_img" not in st.session_state:
        st.session_state.bike_img = None
    if "bike_img_name" not in st.session_state:
        st.session_state.bike_img_name = ""

    uploaded = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded:
        st.session_state.bike_img = Image.open(uploaded)
        st.session_state.bike_img_name = uploaded.name

    if st.session_state.bike_img is not None:
        st.image(st.session_state.bike_img, caption=st.session_state.bike_img_name,
                 use_container_width=True)
        if st.button("🔍 Predict", key="bike_predict"):
            with st.spinner("กำลังวิเคราะห์..."):
                try:
                    arr = preprocess_image(st.session_state.bike_img)
                    pred = bike_model.predict(arr)
                    result_idx = int(np.argmax(pred))
                    result_label = BIKE_CLASS_NAMES[result_idx]
                    status = "ALLOWED" if result_label == "big_bike_500cc" else "NOT ALLOWED"
                    if status == "ALLOWED":
                        st.success(f"**Status: {status}** — Class: {result_label}")
                    else:
                        st.error(f"**Status: {status}** — Class: {result_label}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════
#  Tab 2 — Gambling Text Detector
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('[ทดสอบโมเดลนี้ คลิกที่นี่ (เลื่อนไปส่วน "ใช้งานโมเดล")](#gambling-use-model)')
    st.subheader("ข้อมูลโมเดล")
    st.markdown("""
### ที่มาและความสำคัญของปัญหา
จากการขยายตัวของแพลตฟอร์มวิดีโอออนไลน์ในประเทศไทย โดยเฉพาะ YouTube พบปัญหาการคุกคามจากกลุ่มมิจฉาชีพที่ใช้ระบบอัตโนมัติ (Botnet) ในการส่งข้อความสแปมเพื่อโฆษณาเว็บพนันออนไลน์ในช่องความคิดเห็นของยูทูบเบอร์ไทยชื่อดังและช่องสำนักข่าวใหญ่ๆ สแปมเหล่านี้มักใช้เทคนิคการเลี่ยงคำ (Obfuscation) เช่น การใช้เครื่องหมาย @ หรือสัญลักษณ์พิเศษแทนตัวอักษรเพื่อหลบเลี่ยงระบบคัดกรองมาตรฐานของ YouTube

### ทฤษฎีของอัลกอริทึมที่ใช้พัฒนา
การศึกษานี้เลือกใช้เทคนิค **Ensemble Learning** ซึ่งเป็นการรวมโมเดลหลายตัวเข้าด้วยกันเพื่อให้ได้ผลการทำนายที่แม่นยำและเสถียรกว่าการใช้โมเดลเดี่ยว โดยประกอบด้วยโมเดล 3 ประเภท:

- **Random Forest (Decision Tree based):** ใช้เทคนิค Bagging โดยการสร้าง Decision Trees จำนวนมากที่เป็นอิสระต่อกัน และใช้ Majority Voting เพื่อตัดสินว่าข้อความเป็นสแปมหรือไม่
- **Gradient Boosting (Boosting based):** ทำงานในลักษณะลำดับ โดยต้นไม้ต้นใหม่จะพยายามเรียนรู้และแก้ไขข้อผิดพลาดจากต้นไม้ต้นก่อนหน้า มีความแม่นยำสูงมาก
- **Logistic Regression (Linear based):** ใช้ฟังก์ชันซิกมอยด์ในการคำนวณความน่าจะเป็น ทำหน้าที่เป็น Baseline ที่ดีในการจำแนก Binary Classification (Spam/Ham)

### ขั้นตอนการพัฒนาโมเดล
1. **การรวบรวมและเตรียมข้อมูล:** นำคอมเมนต์มาลบอักขระพิเศษ และทำ Normalization เพื่อเปลี่ยนคำพรางให้เป็นคำมาตรฐาน
2. **การตัดคำและสกัดคุณลักษณะ:** ใช้ไลบรารี **PyThaiNLP** ในการตัดคำภาษาไทย และแปลงข้อความเป็นตัวเลขด้วยวิธี **TF-IDF**
3. **การฝึกฝนแบบ Ensemble:** นำข้อมูลเข้าฝึกฝนโมเดลทั้ง 3 ตัวพร้อมกัน และใช้เทคนิค **Voting Classifier** เพื่อรวมผลลัพธ์
4. **การวัดผล:** ทดสอบประสิทธิภาพด้วย Accuracy และ F1-score

### แหล่งที่มาของชุดข้อมูล
- **UCI Machine Learning Repository:** ชุดข้อมูล YouTube Spam Collection Data Set — คอมเมนต์จริงกว่า 1,956 ข้อความ
- **YouTube Data API v3:** ดึงคอมเมนต์จริงจากช่องยูทูบเบอร์ไทยและสำนักข่าวไทย

### แหล่งอ้างอิง
- UCI Machine Learning Repository: YouTube Spam Collection
- PyThaiNLP: Thai Natural Language Processing Library
- สถาบันวิจัยปัญญาประดิษฐ์ประเทศไทย (AIResearch) — WangchanBERTa
""")

    st.divider()

    st.subheader("ใช้งานโมเดลและผลลัพธ์", anchor="gambling-use-model")

    text_input = st.text_area(
        "ข้อความที่ต้องการตรวจสอบ",
        height=120,
        key="gambling_text_area",
        placeholder="พิมพ์ข้อความที่นี่..."
    )

    if st.button("🔍 Predict", key="gamble_predict"):
        if not text_input.strip():
            st.warning("กรุณาใส่ข้อความ")
        else:
            with st.spinner("กำลังวิเคราะห์..."):
                try:
                    pred = gambling_model.predict([text_input.strip()])[0]
                    result = "Gambling" if pred == 1 else "Clean"
                    if result == "Gambling":
                        st.error(f"ผลลัพธ์: เป็นข้อความมีเกี่ยวข้องกับการพนัน — **{result}** ")
                    else:
                        st.success(f"ผลลัพธ์: เป็นข้อความที่ไม่มีเกี่ยวข้องกับการพนัน — **{result}** ")
                except Exception as e:
                    st.error(f"Error: {e}")

# ─── Footer ────────────────────────────────────────────────
st.divider()
st.caption("Intelligent Systems — 6704063611361")
