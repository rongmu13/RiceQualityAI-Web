# app.py（日本語UI版）
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage import morphology, measure, segmentation, feature

st.set_page_config(page_title="米品質判定AI", layout="wide")
st.title("米品質判定AI（オンライン版）")

with st.sidebar:
    st.markdown("### 手順（左から順に）")
    bg_bright = st.checkbox("背景は明るい（白背景）", value=False)
    th_offset = st.slider("二値化オフセット", -40, 40, 0)
    peak_min  = st.slider("ピーク感度（距離しきい）", 2, 20, 5)
    peak_gap  = st.slider("ピーク間隔（px）", 2, 20, 6)
    min_area  = st.slider("最小粒面積（px）", 20, 600, 80, step=10)
    max_area  = st.slider("最大粒面積（px）", 300, 8000, 3000, step=50)
    learn_btn = st.button("① 学習")
    judge_btn = st.button("② 判定")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("① 健康粒（参照）")
    healthy_file = st.file_uploader("画像をアップロード", type=["jpg","jpeg","png"], key="healthy")
with c2:
    st.subheader("② 白未熟（参照）")
    white_file = st.file_uploader("画像をアップロード", type=["jpg","jpeg","png"], key="white")
with c3:
    st.subheader("③ 判定対象")
    target_file = st.file_uploader("画像をアップロード", type=["jpg","jpeg","png"], key="target")

def read_bgr(file):
    data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def preprocess(img_bgr, bg_bright=False, th_offset=0, peak_min=5, peak_gap=6,
               min_area=80, max_area=3000):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thr = int(np.clip(otsu + th_offset, 0, 255))
    if bg_bright:
        _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

    bw = (bw > 0).astype(np.uint8)
    bw = cv2.erode(bw, np.ones((3,3), np.uint8), 1)
    bw = cv2.dilate(bw, np.ones((3,3), np.uint8), 2)
    bw = cv2.erode(bw, np.ones((3,3), np.uint8), 1)

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    peaks = feature.peak_local_max(dist, min_distance=max(1, int(peak_gap)),
                                   threshold_abs=float(peak_min), labels=bw)

    # 分水嶺
    markers = np.zeros_like(bw, dtype=np.int32)
    for i, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = i
    labels = segmentation.watershed(-dist, markers=markers, mask=bw, connectivity=2)

    props = measure.regionprops(labels)
    comps = []
    for p in props:
        a = p.area
        if a < min_area or a > max_area: 
            continue
        minr, minc, maxr, maxc = p.bbox
        comps.append({
            "label": int(p.label),
            "area":  int(a),
            "bbox":  (int(minc), int(minr), int(maxc), int(maxr)),  # x1,y1,x2,y2
            "coords": p.coords
        })
    return bw, labels, comps

def whiteness_score(img_bgr, comp):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:,:,1] / 255.0
    v = hsv[:,:,2] / 255.0
    yy, xx = comp["coords"][:,0], comp["coords"][:,1]
    vals = v[yy, xx] - 0.7 * s[yy, xx]  # 明度 − 0.7*彩度
    return float(vals.mean()) if vals.size else 0.0

def draw_boxes(img_bgr, comps, is_white=None):
    out = img_bgr.copy()
    for i, comp in enumerate(comps):
        x1, y1, x2, y2 = comp["bbox"]
        if is_white is None:
            color = (106,161,255)  # 青
        else:
            color = (91,91,255) if is_white[i] else (53,211,57)  # 赤/緑
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    return out

def run_one(file):
    img = read_bgr(file)
    bw, labels, comps = preprocess(
        img, bg_bright=bg_bright, th_offset=th_offset,
        peak_min=peak_min, peak_gap=peak_gap,
        min_area=min_area, max_area=max_area
    )
    return img, bw, labels, comps

# 参照画像の処理
h = w = t = None
h_img = w_img = t_img = None
h_comps = w_comps = t_comps = []

if healthy_file:
    h_img, h_bw, h_labels, h_comps = run_one(healthy_file)
    st.image(cv2.cvtColor(draw_boxes(h_img, h_comps), cv2.COLOR_BGR2RGB),
             caption=f"健康粒：{len(h_comps)} 粒を検出", use_container_width=True)

if white_file:
    w_img, w_bw, w_labels, w_comps = run_one(white_file)
    st.image(cv2.cvtColor(draw_boxes(w_img, w_comps), cv2.COLOR_BGR2RGB),
             caption=f"白未熟：{len(w_comps)} 粒を検出", use_container_width=True)

if target_file:
    t_img, t_bw, t_labels, t_comps = run_one(target_file)
    st.image(cv2.cvtColor(draw_boxes(t_img, t_comps), cv2.COLOR_BGR2RGB),
             caption=f"判定対象：{len(t_comps)} 粒を検出", use_container_width=True)

def learn_threshold(h_img, h_comps, w_img, w_comps):
    if not h_comps or not w_comps:
        return None
    hs = [whiteness_score(h_img, c) for c in h_comps]
    ws = [whiteness_score(w_img, c) for c in w_comps]
    th = float((np.median(hs) + np.median(ws)) / 2.0)  # 中点
    return th

if learn_btn:
    if h_img is None or w_img is None:
        st.warning("①健康粒 と ②白未熟 の参照画像を先にアップロードしてください。")
    else:
        th = learn_threshold(h_img, h_comps, w_img, w_comps)
        if th is None:
            st.warning("参照画像から有効な粒を検出できませんでした。パラメータを調整して「一枠一粒」にしてください。")
        else:
            st.session_state["threshold"] = th
            st.success(f"学習完了：しきい値 = {th:.4f}")

if judge_btn:
    if t_img is None or not t_comps:
        st.warning("③判定対象 をアップロードし、「一枠一粒」になるよう調整してください。")
    elif "threshold" not in st.session_state:
        st.warning("先に ①学習 を実行してください。")
    else:
        th = float(st.session_state["threshold"])
        scores = [whiteness_score(t_img, c) for c in t_comps]
        is_white = [s >= th for s in scores]
        total = len(is_white)
        n_white = int(np.sum(is_white))
        rate = (100.0 * n_white / total) if total else 0.0

        vis = draw_boxes(t_img, t_comps, is_white=is_white)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                 caption=f"結果：総数 {total}｜白未熟 {n_white}｜割合 {rate:.1f}%", use_container_width=True)

        df = pd.DataFrame({
            "id": np.arange(1, total+1, dtype=int),
            "area": [c["area"] for c in t_comps],
            "score": [float(s) for s in scores],
            "class": ["白未熟" if f else "整粒" for f in is_white]
        })
        st.dataframe(df, use_container_width=True)
        st.download_button("CSVをダウンロード",
                           data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="result.csv", mime="text/csv")

if not (healthy_file or white_file or target_file):
    st.info("画像をアップロードし、左側のパラメータで「一枠一粒」の状態に調整してください。完了後：①学習 → ②判定。")
