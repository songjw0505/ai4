# streamlit_app.py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="축구팀 이미지 분류기", page_icon="⚽", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ 축구팀 이미지 분류기")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드 (Drive 또는 GitHub Raw URL 사용 가능)
# ======================
MODEL_URL = st.secrets.get("MODEL_URL", "")  # GitHub Raw URL 권장
MODEL_PATH = "soccer_model.pkl"

@st.cache_resource
def load_model(url: str, output_path: str):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model(MODEL_URL, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 축구팀:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨별 콘텐츠
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    "FC Barcelona": {
        "texts": ["스페인 라리가 소속", "홈 구장: 캄프 누", "대표 선수: 리오넬 메시 등"],
        "images": ["https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg"],
        "videos": []
    },
    "Real Madrid": {
        "texts": ["스페인 라리가 소속", "홈 구장: 산티아고 베르나베우", "대표 선수: 크리스티아누 호날두 등"],
        "images": ["https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg"],
        "videos": []
    },
    "Manchester United": {
        "texts": ["잉글리시 프리미어리그 소속", "홈 구장: 올드 트래포드", "대표 선수: 브루노 페르난데스 등"],
        "images": ["https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg"],
        "videos": []
    },
    "Liverpool FC": {
        "texts": ["잉글리시 프리미어리그 소속", "홈 구장: 안필드", "대표 선수: 모하메드 살라 등"],
        "images": ["https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg"],
        "videos": []
    },
    "Chelsea FC": {
        "texts": ["잉글리시 프리미어리그 소속", "홈 구장: 스탬포드 브리지", "대표 선수: 은골로 캉테 등"],
        "images": ["https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg"],
        "videos": []
    }
}

# ======================
# 유틸 함수
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력: 카메라 / 파일 업로드
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라", "📁 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam: new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지 업로드 (jpg, png, jpeg, webp, tiff)", type=["jpg","png","jpeg","webp","tiff"])
    if f: new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1,1], vertical_alignment="center")
    pil_img = load_pil_from_bytes(st.session_state.img_bytes)

    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(f"""
        <div class="prediction-box">
            <span style="font-size:1rem;color:#555;">예측 결과:</span>
            <h2>{st.session_state.last_prediction}</h2>
            <div class="helper">오른쪽 패널에서 라벨별 정보가 표시됩니다.</div>
        </div>
        """, unsafe_allow_html=True)

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted([(labels[i], float(probs[i])) for i in range(len(labels))],
                           key=lambda x: x[1], reverse=True)
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(f"""
            <div class="prob-card">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <strong>{lbl}</strong><span>{pct:.2f}%</span>
              </div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # 오른쪽: 라벨별 콘텐츠
    with right:
        st.subheader("라벨별 정보")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 팀 선택", labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 정보가 없습니다. 코드에서 추가하세요.")
        else:
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f'<div class="card" style="grid-column:span 12;"><h4>설명</h4>{t}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f'<div class="card" style="grid-column:span 4;"><h4>이미지</h4><img src="{url}" class="thumb"/></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    st.markdown(f'<div class="card" style="grid-column:span 6;"><h4>동영상</h4><a href="{v}" target="_blank">{v}</a></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("카메라 촬영 또는 파일 업로드 후 분석 결과와 팀 정보를 확인할 수 있습니다.")
