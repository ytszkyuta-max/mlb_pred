"""
MLB 投球予測 AI — Streamlit UI (Ensemble版)
起動: streamlit run mlb_pred/app.py
"""
import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from catboost import Pool

MPH_TO_KMH = 1.60934

PITCH_JP = {
    "FF": "速球(4シーム)", "SI": "シンカー",    "SL": "スライダー",
    "CH": "チェンジアップ", "CU": "カーブ",     "FC": "カットボール",
    "FS": "スプリット",    "ST": "スイーパー",   "KC": "ナックルカーブ",
}
PITCH_COLOR = {
    "FF": "#e74c3c", "SI": "#e67e22", "SL": "#27ae60", "CH": "#2980b9",
    "CU": "#8e44ad", "FC": "#16a085", "FS": "#f39c12", "ST": "#2ecc71",
    "KC": "#7f8c8d",
}
PREV_OPTS = {
    "なし（初球）": 0,
    "FF 速球(4シーム)": 1, "SI シンカー": 2,    "SL スライダー": 3,
    "CH チェンジアップ": 4, "CU カーブ": 5,    "FC カットボール": 6,
    "FS スプリット": 7,    "ST スイーパー": 8, "KC ナックルカーブ": 9,
}


@st.cache_resource
def load_model():
    path = Path(__file__).parent / "models" / "ensemble_stack.joblib"
    if not path.exists():
        return None
    d = joblib.load(path)
    return (
        d["meta_model"], d["xgb_model"], d["cat_model"],
        d["le_target"], d["le_pitcher"], d["cat_feature_indices"],
    )


def ensemble_predict_proba(meta_model, xgb_model, cat_model, cat_feature_indices, X):
    """XGBoost + CatBoost の確率をスタッキングしてメタ学習器で予測する。"""
    xgb_probs = xgb_model.predict_proba(X)

    X_obj = X.astype(object)
    for idx in cat_feature_indices:
        X_obj[:, idx] = X[:, idx].astype(int).astype(str)
    cat_probs = cat_model.predict_proba(Pool(X_obj, cat_features=cat_feature_indices))

    return meta_model.predict_proba(np.hstack([xgb_probs, cat_probs]))


def render_diamond(on_1b: bool, on_2b: bool, on_3b: bool) -> str:
    d2 = "🟡" if on_2b else "⬜"
    d1 = "🟡" if on_1b else "⬜"
    d3 = "🟡" if on_3b else "⬜"
    return (
        f'<div style="font-size:1.8rem; text-align:center; line-height:1.5; margin:8px 0;">'
        f"&nbsp;&nbsp;{d2}<br>"
        f"{d3}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{d1}<br>"
        f"&nbsp;&nbsp;🏠</div>"
    )


def probability_bar(cls: str, prob: float) -> str:
    jp    = PITCH_JP.get(cls, cls)
    color = PITCH_COLOR.get(cls, "#888")
    width = max(2, int(prob * 220))
    return (
        f'<div style="margin:5px 0; display:flex; align-items:center; gap:8px">'
        f'  <span style="width:165px; font-size:.88rem">{jp}&nbsp;({cls})</span>'
        f'  <div style="width:{width}px; height:18px; background:{color};'
        f'    border-radius:4px; flex-shrink:0"></div>'
        f'  <span style="font-size:.88rem; color:#444">{prob*100:.1f}%</span>'
        f"</div>"
    )


def main():
    st.set_page_config(page_title="MLB 投球予測 AI", page_icon="⚾", layout="wide")
    st.title("⚾ MLB 投球予測 AI")
    st.caption("試合状況を入力して次の投球を予測しよう！（XGBoost + CatBoost アンサンブル）")

    loaded = load_model()
    if loaded is None:
        st.error(
            "`python main.py --mode ensemble` でモデルを先に学習してから起動してください。"
        )
        st.stop()

    meta_model, xgb_model, cat_model, le_target, le_pitcher, cat_feature_indices = loaded

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    col_in, col_out = st.columns([1, 1], gap="large")

    # ════════════════════════════════════════════════════════════
    # 左パネル: 試合状況の入力
    # ════════════════════════════════════════════════════════════
    with col_in:
        st.subheader("📋 試合状況")

        c1, c2 = st.columns(2)
        with c1:
            p_throws = st.radio("投手の利き腕", ["右投げ", "左投げ"], horizontal=True)
        with c2:
            stand = st.radio("打者の打席", ["右打ち", "左打ち"], horizontal=True)
        p_throws_val = 1 if p_throws == "右投げ" else 0
        stand_val    = 1 if stand    == "右打ち" else 0

        st.write("")

        c1, c2, c3 = st.columns(3)
        with c1:
            balls   = st.number_input("ボール",    0, 3, 0)
        with c2:
            strikes = st.number_input("ストライク", 0, 2, 0)
        with c3:
            outs    = st.number_input("アウト",    0, 2, 0)

        b_vis = "● " * balls   + "○ " * (3 - balls)
        s_vis = "● " * strikes + "○ " * (2 - strikes)
        o_vis = "✕ " * outs    + "○ " * (2 - outs)
        st.markdown(f"`B {b_vis}  S {s_vis}  O {o_vis}`")

        inning = st.slider("イニング", 1, 12, 1)

        st.write("**ランナー状況**")
        r1, r2, r3 = st.columns(3)
        with r1: on_1b = st.checkbox("一塁")
        with r2: on_2b = st.checkbox("二塁")
        with r3: on_3b = st.checkbox("三塁")
        st.markdown(render_diamond(on_1b, on_2b, on_3b), unsafe_allow_html=True)

        st.divider()

        prev_label = st.selectbox("直前の球種", list(PREV_OPTS.keys()))
        prev_val   = PREV_OPTS[prev_label]

        # 球速: ユーザーは km/h で入力 → モデルには mph で渡す
        prev_speed_kmh = st.slider(
            "直前の球速 (km/h)",
            min_value=96, max_value=175,
            value=148, step=1,
            disabled=(prev_val == 0),
        )
        if prev_val != 0:
            st.caption(f"≈ {prev_speed_kmh / MPH_TO_KMH:.1f} mph")
        prev_speed_mph = prev_speed_kmh / MPH_TO_KMH

        pitcher_enc = int(np.median(np.arange(len(le_pitcher.classes_))))

        if st.button("🎯 投球を予測する！", type="primary", use_container_width=True):
            X = np.array([[
                balls, strikes, outs, inning,
                stand_val, p_throws_val,
                int(on_1b), int(on_2b), int(on_3b),
                prev_val, prev_speed_mph,
                pitcher_enc,
            ]], dtype=float)

            proba = ensemble_predict_proba(
                meta_model, xgb_model, cat_model, cat_feature_indices, X
            )[0]
            best_i = int(proba.argmax())
            st.session_state.last_result = {
                "proba":    proba,
                "best_cls": le_target.classes_[best_i],
            }

    # ════════════════════════════════════════════════════════════
    # 右パネル: 予測結果
    # ════════════════════════════════════════════════════════════
    with col_out:
        st.subheader("📊 予測結果")

        result = st.session_state.last_result
        if result is None:
            st.markdown(
                '<div style="color:#999; text-align:center; margin-top:80px; font-size:1.1rem">'
                "← 左のパネルで試合状況を入力して<br>「予測する」ボタンを押してください"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            proba    = result["proba"]
            best_cls = result["best_cls"]
            best_jp  = PITCH_JP.get(best_cls, best_cls)
            best_p   = float(proba.max())

            st.success(
                f"### ⚾ {best_jp}（{best_cls}）\n"
                f"**予測確率: {best_p*100:.1f}%**"
            )

            st.write("#### 各球種の確率")
            for i in np.argsort(proba)[::-1]:
                st.markdown(
                    probability_bar(le_target.classes_[i], float(proba[i])),
                    unsafe_allow_html=True,
                )

            top3 = np.argsort(proba)[::-1][:3]
            summary = " ＞ ".join(
                f"{PITCH_JP.get(le_target.classes_[i], le_target.classes_[i])}"
                f" {proba[i]*100:.0f}%"
                for i in top3
            )
            st.info(f"📊 上位3: {summary}")

            st.divider()

            with st.form("record_form"):
                st.write("**実際に投げられた球種を記録（任意）**")
                actual_opts = ["— スキップ —"] + [
                    f"{cls}　{PITCH_JP.get(cls, cls)}"
                    for cls in le_target.classes_
                ]
                actual_sel = st.selectbox("実際の球種", actual_opts, label_visibility="collapsed")
                if st.form_submit_button("記録する"):
                    if actual_sel != "— スキップ —":
                        actual_cls = actual_sel.split()[0]
                        correct = actual_cls == best_cls
                        st.session_state.history.append(correct)
                        if correct:
                            st.balloons()
                            st.success("🎉 正解！")
                        else:
                            st.warning(
                                f"残念！実際は {PITCH_JP.get(actual_cls, actual_cls)}"
                                f"（{actual_cls}）でした。"
                            )

            h = st.session_state.history
            if h:
                acc = sum(h) / len(h)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("予測正解率", f"{acc*100:.0f}%")
                with c2:
                    st.metric("結果", f"{sum(h)} / {len(h)} 正解")
                if st.button("記録をリセット", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()


if __name__ == "__main__":
    main()
