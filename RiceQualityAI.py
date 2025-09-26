# RiceQualityAI.py
import streamlit as st, numpy as np, pandas as pd, cv2
from skimage import measure, segmentation, feature

st.set_page_config(page_title="米品質判定AI", layout="wide")
st.title("米品質判定AI")

with st.sidebar:
    st.markdown("### 調整 → 学習 → 判定")
    bg_bright = st.checkbox("背景は明るい（白背景）", value=False)
    th_offset = st.slider("二値化オフセット", -40, 40, 0)
    peak_gap  = st.slider("ピーク間隔（px）", 2, 20, 6)
    min_area  = st.slider("最小粒面積（px）", 20, 600, 80, 10)
    max_area  = st.slider("最大粒面積（px）", 300, 8000, 3000, 50)
    learn_btn = st.button("学習")
    judge_btn = st.button("判定")

up = st.file_uploader("画像をアップロード（JPG/PNG）", type=["jpg","jpeg","png"])

def read_img(f):
    arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def segment(img, bg_bright, th_offset, peak_gap, min_area, max_area):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray); gray = cv2.GaussianBlur(gray,(3,3),0)
    ret, _ = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thr = int(np.clip(ret + th_offset, 0,255))
    mode = cv2.THRESH_BINARY_INV if bg_bright else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, thr, 255, mode)
    bw = cv2.erode(bw, np.ones((3,3),np.uint8),1)
    bw = cv2.dilate(bw, np.ones((3,3),np.uint8),2)
    bw = cv2.erode(bw, np.ones((3,3),np.uint8),1)
    dist = cv2.distanceTransform((bw>0).astype(np.uint8), cv2.DIST_L2, 5)
    peaks = feature.peak_local_max(dist, min_distance=max(1,int(peak_gap)),
                                   labels=(bw>0))
    markers = np.zeros_like(bw, np.int32)
    for i,(y,x) in enumerate(peaks,1): markers[y,x]=i
    labels = segmentation.watershed(-dist, markers=markers, mask=(bw>0))
    comps = []
    for p in measure.regionprops(labels):
        a=p.area
        if a<min_area or a>max_area: continue
        minr,minc,maxr,maxc = p.bbox
        comps.append({"area":int(a),"bbox":(minc,minr,maxc,maxr),"coords":p.coords})
    return bw, comps

def whiteness(img, comp):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:,:,1]/255.0; v = hsv[:,:,2]/255.0
    yy,xx = comp["coords"][:,0], comp["coords"][:,1]
    vals = v[yy,xx] - 0.7*s[yy,xx]
    return float(vals.mean()) if vals.size else 0.0

def draw(img, comps, flags=None):
    out = img.copy()
    for i,c in enumerate(comps):
        x1,y1,x2,y2 = c["bbox"]
        col = (91,91,255) if (flags and flags[i]) else (53,211,57)
        cv2.rectangle(out,(x1,y1),(x2,y2),col,2,cv2.LINE_AA)
    return out

if up:
    img = read_img(up)
    bw, comps = segment(img, bg_bright, th_offset, peak_gap, min_area, max_area)
    col1,col2 = st.columns(2)
    with col1: st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原画像", use_container_width=True)
    with col2: st.image(bw, caption="二値/分割（「一枠一粒」へ調整）", use_container_width=True)

    # 学習
    if learn_btn:
        if not comps: st.warning("粒が検出されていません。"); 
        else:
            areas = np.array([c["area"] for c in comps])
            st.session_state["th_area"] = float(np.quantile(areas, 0.2))
            st.success(f"学習完了：面積しきい値 = {st.session_state['th_area']:.1f}")

    # 判定
    if judge_btn:
        if "th_area" not in st.session_state: st.warning("先に「学習」を押してください。")
        else:
            th = st.session_state["th_area"]
            flags = [c["area"]<th for c in comps]  # True=小粒（例：白未熟）
            vis = draw(img, comps, flags)
            total = len(flags); n_bad = int(np.sum(flags)); rate = 100*n_bad/total if total else 0
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                     caption=f"総数 {total}｜小粒(例:白未熟) {n_bad}｜割合 {rate:.1f}%",
                     use_container_width=True)
            df = pd.DataFrame({"id":np.arange(1,total+1),"area":[c['area'] for c in comps],
                               "class":["白未熟" if f else "整粒" for f in flags]})
            st.download_button("CSVダウンロード", data=df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="result.csv", mime="text/csv")
else:
    st.info("画像をアップロードし、左側のパラメータで「一枠一粒」に調整。完了後：「学習」→「判定」。")
