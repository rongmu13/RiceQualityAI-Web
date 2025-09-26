import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage import measure, segmentation, feature

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except Exception:
    LogisticRegression = None
    SKLEARN_OK = False

st.set_page_config(page_title="米品質判定AI.Shinshu-U R.Y.", layout="wide")
st.title("米品質判定AI")

#显示尺寸
W_COL    = 360   # 第一行
W_RESULT = 900   # 第二行

#菜单
with st.sidebar:
    st.markdown("### 調整 → 学習 → 判定")
    bg_bright = st.checkbox("背景は明るい（白背景）", value=False)
    th_offset = st.slider("二値化オフセット", -40, 40, 0)
    peak_min  = st.slider("ピーク感度", 2, 20, 5)
    peak_gap  = st.slider("ピーク間隔（px）", 2, 20, 6)
    min_area  = st.slider("最小粒面積（px）", 20, 600, 80, step=10)
    max_area  = st.slider("最大粒面積（px）", 300, 8000, 3000, step=50)
    open_iter = st.slider("分離の強さ", 0, 3, 1)
    st.markdown("---")
    use_ml = st.checkbox("軽量MLを使う",
                         value=False, disabled=not SKLEARN_OK,
                         help=None if SKLEARN_OK else "scikit-learnが無い環境なのでオフのまま使ってください")
    learn_btn = st.button("① 学習")
    judge_btn = st.button("② 判定")

#上传
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("① 健康粒（参照）")
    healthy_file = st.file_uploader("画像をアップロード・同条件で撮影した画像を使用してください", type=["jpg","jpeg","png"], key="healthy")
with c2:
    st.subheader("② 白未熟（参照）")
    white_file = st.file_uploader("画像をアップロード・同条件で撮影した画像を使用してください", type=["jpg","jpeg","png"], key="white")
with c3:
    st.subheader("③ 判定対象")
    target_file = st.file_uploader("画像をアップロード・同条件で撮影した画像を使用してください", type=["jpg","jpeg","png"], key="target")

#共通函数
def read_bgr(uploaded):
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def preprocess(img_bgr, bg_bright, th_offset, peak_min, peak_gap, min_area, max_area, open_iter):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(np.clip(ret + th_offset, 0, 255))
    mode = cv2.THRESH_BINARY_INV if bg_bright else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, thr, 255, mode)
    bw = (bw > 0).astype(np.uint8)

    k3 = np.ones((3,3), np.uint8)
    if open_iter > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=int(open_iter))
    else:
        bw = cv2.erode(bw, k3, 1); bw = cv2.dilate(bw, k3, 2); bw = cv2.erode(bw, k3, 1)

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    peaks = feature.peak_local_max(dist, min_distance=max(1,int(peak_gap)),
                                   threshold_abs=float(peak_min), labels=bw)
    if len(peaks) < 3:
        peaks = feature.peak_local_max(dist, min_distance=max(1,int(max(1,peak_gap//2))), labels=bw)

    markers = np.zeros_like(bw, np.int32)
    for i,(y,x) in enumerate(peaks,1): markers[y,x] = i
    if markers.max() == 0:
        labels = measure.label(bw, connectivity=2)
    else:
        labels = segmentation.watershed(-dist, markers=markers, mask=bw)

    comps = []
    for p in measure.regionprops(labels):
        a = int(p.area)
        if a < min_area or a > max_area: continue
        minr, minc, maxr, maxc = p.bbox
        comps.append({"area": a, "bbox": (int(minc), int(minr), int(maxc), int(maxr)), "coords": p.coords})
    return bw*255, comps

def whiteness_score(img_bgr, comp):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:,:,1]/255.0; v = hsv[:,:,2]/255.0
    yy, xx = comp["coords"][:,0], comp["coords"][:,1]
    vals = v[yy,xx] - 0.7*s[yy,xx]
    return float(vals.mean()) if vals.size else 0.0

def draw_boxes(img_bgr, comps, flags=None):
    out = img_bgr.copy()
    for i,c in enumerate(comps):
        x1,y1,x2,y2 = c["bbox"]
        col = (106,161,255) if flags is None else ((91,91,255) if flags[i] else (53,211,57))
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
    return out

#图像分割
h_img = w_img = t_img = None
h_comps = w_comps = t_comps = []

if healthy_file:
    h_img = read_bgr(healthy_file)
    _, h_comps = preprocess(h_img, bg_bright, th_offset, peak_min, peak_gap, min_area, max_area, open_iter)

if white_file:
    w_img = read_bgr(white_file)
    _, w_comps = preprocess(w_img, bg_bright, th_offset, peak_min, peak_gap, min_area, max_area, open_iter)

if target_file:
    t_img = read_bgr(target_file)
    _, t_comps = preprocess(t_img, bg_bright, th_offset, peak_min, peak_gap, min_area, max_area, open_iter)

#第一行
with c1:
    if h_img is not None:
        st.image(cv2.cvtColor(draw_boxes(h_img, h_comps), cv2.COLOR_BGR2RGB),
                 caption=f"健康粒：{len(h_comps)} 粒", width=W_COL)

with c2:
    if w_img is not None:
        st.image(cv2.cvtColor(draw_boxes(w_img, w_comps), cv2.COLOR_BGR2RGB),
                 caption=f"白未熟：{len(w_comps)} 粒", width=W_COL)

with c3:
    if t_img is not None:
        st.image(cv2.cvtColor(draw_boxes(t_img, t_comps), cv2.COLOR_BGR2RGB),
                 caption=f"判定対象：{len(t_comps)} 粒（判定前）", width=W_COL)

#学習
def learn_threshold(h_img, h_comps, w_img, w_comps):
    if not h_comps or not w_comps: return None
    hs = [whiteness_score(h_img, c) for c in h_comps]
    ws = [whiteness_score(w_img, c) for c in w_comps]
    return float((np.median(hs) + np.median(ws)) / 2.0)

def learn_ml(h_img, h_comps, w_img, w_comps):
    if (not SKLEARN_OK) or (LogisticRegression is None): return None
    if not h_comps or not w_comps: return None
    X, y = [], []
    for c in h_comps: X.append([whiteness_score(h_img, c), float(c["area"])]); y.append(0)
    for c in w_comps: X.append([whiteness_score(w_img, c), float(c["area"])]); y.append(1)
    if len(X) < 10: return None
    clf = LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y))
    return clf

if learn_btn:
    if h_img is None or w_img is None:
        st.warning("①健康粒 と ②白未熟 の参照画像を先にアップロードしてください。")
    else:
        th = learn_threshold(h_img, h_comps, w_img, w_comps)
        if th is None:
            st.warning("参照画像から有効な粒を検出できませんでした。『一枠一粒』に調整してください。")
        else:
            st.session_state["threshold"] = th
            msg = f"学習完了：しきい値 = {th:.4f}"
            if use_ml and SKLEARN_OK:
                clf = learn_ml(h_img, h_comps, w_img, w_comps)
                if clf is not None:
                    st.session_state["clf"] = clf; msg += " ｜ ML=有効"
                else:
                    st.session_state.pop("clf", None); msg += " ｜ ML=無効（参照粒が少ない）"
            st.success(msg)

#判定
if judge_btn:
    if t_img is None or not t_comps:
        st.warning("③判定対象 をアップロードし、『一枠一粒』に調整してください。")
    elif "threshold" not in st.session_state:
        st.warning("先に ①学習 を実行してください。")
    else:
        th = float(st.session_state["threshold"])
        clf = st.session_state.get("clf", None) if (use_ml and SKLEARN_OK) else None

        scores = [whiteness_score(t_img, c) for c in t_comps]
        if clf is not None:
            Xtest = np.stack([scores, [float(c["area"]) for c in t_comps]], axis=1)
            prob = clf.predict_proba(Xtest)[:,1]
            is_white = prob >= 0.5
        else:
            is_white = np.array([s >= th for s in scores])

        total = len(is_white); n_white = int(np.sum(is_white))
        rate = (100.0 * n_white / total) if total else 0.0

        vis_big = cv2.cvtColor(
            draw_boxes(t_img, t_comps, flags=is_white.tolist()),
            cv2.COLOR_BGR2RGB
        )

        st.divider()
        st.subheader("判定結果")
        st.image(vis_big, width=W_RESULT,
                 caption=f"結果：総数 {total}｜白未熟 {n_white}｜割合 {rate:.1f}%"
                         + (" ｜ ML使用" if clf is not None else " ｜ しきい値使用"))

        df = pd.DataFrame({
            "id": np.arange(1, total+1, dtype=int),
            "area": [int(c["area"]) for c in t_comps],
            "score": [float(s) for s in scores],
            "class": ["白未熟" if f else "整粒" for f in is_white]
        })
        st.dataframe(df, use_container_width=True)
        st.download_button("CSVをダウンロード",
                           data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="result.csv", mime="text/csv")

#初期
if not (healthy_file or white_file or target_file):
    st.info("画像をアップロードし、左側のパラメータで『一枠一粒』に調整。完了後：①学習 → ②判定。")



