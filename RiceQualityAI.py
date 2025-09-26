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
st.title("米品質判定AI　Shinshu Univ. R.Y.")


#使い方ガイド
with st.expander("📖 使い方ガイド", expanded=False):
    st.markdown("""
**目的**：青い枠ひとつに米粒がちょうど一粒収まるように調整する。

**おすすめ順：**  
1）背景を正しく設定 →  
2）二値化オフセットで輪郭調整 →  
3）ピーク感度 / ピーク間隔 + 分離の強さ で「1粒=1中心点」 →  
4）最小/最大粒面積で外れ値カット

---

### 1) 明るい背景
- 白い紙・ライトボックスで撮影 → **チェック**  
- 黒い台・暗い背景 → **チェックを外す**  
※ ここが違うと他の調整が効きにくいです。

### 2) 二値化オフセット
- 目的：米粒の **輪郭の太さ** を微調整  
- 動かし方：**±5〜15** を少しずつ  
- 白背景＝**オン**：**＋で太く**／**−で細く**  
- 白背景＝**オフ**：**＋で細く**／**−で太く**  
- 困ったら：**欠ける → −側**／**にじむ → ＋側**

### 3) ピーク感度
- 意味：粒を **1つとして数える厳しさ**（高いほど厳しい）  
- 値↑ → まとまりやすい（分割が減る）／ 値↓ → 割れやすい（過分割）  
- 目安：**4〜10**（画面で見た米粒の**短い辺の半分**くらい）

### 4) ピーク間隔
- 意味：粒の **中心点どうしの最小距離**  
- 値↑ → 近い粒が 1 つに見えやすい ／ 値↓ → 過分割になりやすい  
- 目安：**5〜12**（**短径と同程度〜やや小さめ**）

### 5) 最小粒面積
- 意味：この面積 **未満** は **ゴミ** として除去  
- 目安：**50〜200**（↑でノイズに強いが小粒も消えやすい）

### 6) 最大粒面積
- 意味：この面積 **超え** は **大きすぎる塊** として除外  
- 目安：**1500〜5000**（解像度に依存。↓で連結塊を弾きやすい）

### 7) 分離の強さ
- 意味：くっついた粒をどの程度強制的に分けるか。
- 値↑ → 強く分離（くっつきを剥がしやすいが，小粒が欠けやすい）
- 値↓ → 分離は弱い（粒が残りやすいが，密集部分は一塊に見えやすい）
- 目安：0〜2 回程度。画像の重なり具合によって調整。

---

### 早見表
- 小粒やゴミまで数える → **最小面積↑ / ピーク感度↑**  
- 近い粒が 1 つにくっつく → **ピーク間隔↓ / ピーク感度↓**  
- 輪郭が欠ける/ギザギザ → **二値化オフセット −側**  
- 太って重なって見える → **二値化オフセット ＋側**

### 初期の目安
- 二値化オフセット：**±10 前後**  
- ピーク感度：**6** ／ ピーク間隔：**8**  
- 最小粒面積：**100** ／ 最大粒面積：**3000**  
- 分離の強さ：1（0〜2 の範囲で調整）
- 背景は撮影条件に合わせてオン/オフ

---

### ⭐ 最後に 
設定が整い『**1枠=1粒**』に近づいたら、モデルを選んで、**①学習** を押し、学習完了後に **②判定** を押してください。
""")




#显示尺寸
W_COL    = 360   # 第一行
W_RESULT = 900   # 第二行

#菜单
with st.sidebar:
    st.markdown("### 調整 → 学習 → 判定")
    bg_bright = st.checkbox("明るい背景", value=False)
    th_offset = st.slider("二値化オフセット", -40, 40, 0)
    peak_min  = st.slider("ピーク感度", 2, 20, 5)
    peak_gap  = st.slider("ピーク間隔", 2, 20, 6)
    min_area  = st.slider("最小粒面積", 20, 600, 80, step=10)
    max_area  = st.slider("最大粒面積", 300, 8000, 3000, step=50)
    open_iter = st.slider("分離の強さ", 0, 3, 1)
    st.markdown("---")

    ml_mode = st.selectbox(
        "学習方法",
        ["しきい値のみ", "Logistic", "CNN"],
        index=0,
        help="参照画像から学習して分類。Logistic=線形分類、CNN=小型畳み込みネット"
    )
    learn_btn = st.button("① 学習")
    judge_btn = st.button("② 判定")



#上传
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("① 健康粒（学習用）")
    healthy_file = st.file_uploader("画像をアップロード・同条件で撮影した画像を使用してください", type=["jpg","jpeg","png"], key="healthy")
with c2:
    st.subheader("② 白未熟（学習用）")
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


#小贴图
def crop_patch(img_bgr, comp, size=48):
    x1,y1,x2,y2 = comp["bbox"]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_bgr.shape[1]-1, x2); y2 = min(img_bgr.shape[0]-1, y2)
    crop = img_bgr[y1:y2+1, x1:x2+1]
    if crop.size == 0:  # 兜底
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

#CNN
import tensorflow as tf

def build_tiny_cnn(input_size=48):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size, input_size, 3)),
        tf.keras.layers.Rescaling(1/255.0),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")  # 1=白未熟
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m

def train_cnn(h_img, h_comps, w_img, w_comps, input_size=48, max_per_class=400, epochs=10, batch=32):
    # 采样（均衡）
    H = h_comps[:max_per_class] if len(h_comps) > max_per_class else h_comps
    W = w_comps[:max_per_class] if len(w_comps) > max_per_class else w_comps
    n = min(len(H), len(W))
    if n < 10:
        return None, "参照の粒子が少なすぎます（各10+推奨）"

    xs, ys = [], []
    for i in range(n):
        xs.append(crop_patch(h_img, H[i], input_size)); ys.append(0)  # 健康=0
        xs.append(crop_patch(w_img, W[i], input_size)); ys.append(1)  # 白未熟=1
    X = np.asarray(xs, dtype=np.uint8)
    y = np.asarray(ys, dtype=np.float32)

    # 打乱
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    model = build_tiny_cnn(input_size)
    h = model.fit(X, y, epochs=epochs, batch_size=batch, validation_split=0.15, verbose=0)
    va = float(h.history["val_accuracy"][-1])
    return model, f"CNN学習完了：Validation accuracy={va*100:.1f}%"

def cnn_predict(model, img_bgr, comps, input_size=48):
    probs = []
    for c in comps:
        patch = crop_patch(img_bgr, c, input_size)
        p = float(model.predict(patch[None, ...], verbose=0)[0][0])  # 0..1
        probs.append(p)
    return np.array(probs)




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
        # 阈值法：始终学习
        th = learn_threshold(h_img, h_comps, w_img, w_comps)
        if th is None:
            st.warning("参照画像から有効な粒を検出できませんでした。『一枠一粒』に調整してください。")
        else:
            st.session_state["threshold"] = float(th)
            msg = f"学習完了：しきい値 = {th:.4f}"

            # ML 任选其一
            if ml_mode == "Logistic":
                try:
                    from sklearn.linear_model import LogisticRegression
                    clf = learn_ml(h_img, h_comps, w_img, w_comps)
                    if clf is not None:
                        st.session_state["clf"] = clf
                        st.session_state.pop("cnn", None)
                        msg += " ｜ ML=Logistic 有効"
                    else:
                        st.session_state.pop("clf", None)
                        msg += " ｜ ML=Logistic 無効（参照粒が少ない）"
                except Exception:
                    st.session_state.pop("clf", None)
                    msg += " ｜ ML=Logistic 使用不可（環境）"

            elif ml_mode == "CNN":
                model, note = train_cnn(h_img, h_comps, w_img, w_comps, input_size=48, epochs=10)
                if model is not None:
                    st.session_state["cnn"] = model
                    st.session_state.pop("clf", None)
                    msg += " ｜ ML=CNN 有効"
                else:
                    st.session_state.pop("cnn", None)
                    msg += " ｜ ML=CNN 無効（参照粒が少ない）"
                st.info(note)

            else:  # しきい値のみ
                st.session_state.pop("clf", None)
                st.session_state.pop("cnn", None)

            st.success(msg)


#判定
if judge_btn:
    if t_img is None or not t_comps:
        st.warning("③判定対象 をアップロードし、『一枠一粒』に調整してください。")
    elif "threshold" not in st.session_state:
        st.warning("先に ①学習 を実行してください。")
    else:
        th = float(st.session_state["threshold"])
        # 优先使用 CNN，其次 Logistic，否则阈值
        if "cnn" in st.session_state and st.session_state["cnn"] is not None and ml_mode == "CNN":
            probs = cnn_predict(st.session_state["cnn"], t_img, t_comps, input_size=48)
            is_white = probs >= 0.5
            used = "CNN"
        elif "clf" in st.session_state and st.session_state["clf"] is not None and ml_mode == "Logistic":
            scores = [whiteness_score(t_img, c) for c in t_comps]
            Xtest = np.stack([scores, [float(c["area"]) for c in t_comps]], axis=1)
            prob = st.session_state["clf"].predict_proba(Xtest)[:,1]
            is_white = prob >= 0.5
            used = "Logistic"
        else:
            scores = [whiteness_score(t_img, c) for c in t_comps]
            is_white = np.array([s >= th for s in scores])
            used = "しきい値"

        total = len(is_white)
        n_white = int(np.sum(is_white))
        rate = (100.0 * n_white / total) if total else 0.0

        vis_big = cv2.cvtColor(
            draw_boxes(t_img, t_comps, flags=is_white.tolist()),
            cv2.COLOR_BGR2RGB
        )

        st.divider()
        st.subheader("判定結果")
        st.image(vis_big, width=900,
                 caption=f"結果：総数 {total}｜白未熟 {n_white}｜割合 {rate:.1f}% ｜ ML={used}")

        # 导出
        out_scores = (probs if used=="CNN" else (prob if used=="Logistic" else scores))
        df = pd.DataFrame({
            "id": np.arange(1, total+1, dtype=int),
            "area": [int(c['area']) for c in t_comps],
            "score": [float(s) for s in out_scores],
            "class": ["白未熟" if f else "整粒" for f in is_white]
        })
        st.dataframe(df, use_container_width=True)
        st.download_button("CSVをダウンロード",
                           data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="result.csv", mime="text/csv")


#初期
if not (healthy_file or white_file or target_file):
    st.info("画像をアップロードし、左側のパラメータで『一枠一粒』に調整。完了後：①学習 → ②判定。")














