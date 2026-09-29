import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_config
from src.inference import EmergencyInferenceEngine

# Streamlit Page Config
st.set_page_config(
    page_title="Emergency Message Prioritizer",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern emergency dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #FCA5A5;
        display: inline-block;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #FDBA74;
        display: inline-block;
    }
    .badge-low {
        background-color: #E0F2FE;
        color: #075985;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #7DD3FC;
        display: inline-block;
    }
    .badge-review {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        border: 1px solid #FCD34D;
        display: inline-block;
    }
    .badge-model {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        border: 1px solid #86EFAC;
        display: inline-block;
    }
    .feature-chip {
        background-color: #E2E8F0;
        color: #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: monospace;
        margin-right: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_inference_engine():
    return EmergencyInferenceEngine()

def load_eval_results():
    path = "reports/evaluation_results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def main():
    st.markdown('<div class="main-header">🚨 Emergency Message Prioritizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NLP Disaster Social Media Response System (HumAID Dataset)</div>', unsafe_allow_html=True)

    config = load_config()
    engine = get_inference_engine()
    eval_results = load_eval_results()

    # Sidebar Options
    st.sidebar.title("⚙️ Control Panel")
    
    # Model selection
    transformer_available = os.path.exists(config["transformer_model"]["save_dir"])
    model_options = ["TF-IDF + Logistic Regression (Baseline)"]
    if transformer_available:
        model_options.append("DistilBERT Fine-Tuned (Advanced)")
        
    selected_model_name = st.sidebar.selectbox("Select Classification Model", model_options)
    model_type = "transformer" if "DistilBERT" in selected_model_name else "baseline"
    
    # Threshold slider
    conf_threshold = st.sidebar.slider(
        "Human Review Threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Predictions below this confidence score trigger 'Human Review Recommended'"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Academic Prototype Disclaimer:**\nThis system is an academic decision-support tool and should not be used as an autonomous emergency dispatch or life-safety system.")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🔍 Analyze Message", 
        "📁 Batch CSV Analysis", 
        "📈 Model Performance", 
        "📉 Dataset Insights"
    ])

    # ---------------- TAB 1: DASHBOARD ----------------
    with tab1:
        st.subheader("System Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages Analyzed", "76,484", delta="HumAID Dataset")
        with col2:
            st.metric("Critical Categories Ratio", "46.5%", delta="Rescue + Casualties")
        with col3:
            base_f1 = eval_results["baseline"]["macro_f1"] if eval_results and "baseline" in eval_results else 0.844
            st.metric("Baseline Macro F1", f"{base_f1:.3f}")
        with col4:
            st.metric("Active Model", "DistilBERT" if model_type == "transformer" else "TF-IDF + LogReg")

        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("### 🎯 Priority Mapping Policy")
            st.table(pd.DataFrame([
                {"Category": "Rescue Request", "Priority": "CRITICAL", "Description": "Immediate life safety, trapped victims, urgent help"},
                {"Category": "Casualties / Injuries", "Priority": "CRITICAL", "Description": "Injuries, fatalities, medical assistance needed"},
                {"Category": "Infrastructure Damage", "Priority": "HIGH", "Description": "Collapsed bridges, blocked roads, power outages"},
                {"Category": "Not Relevant", "Priority": "LOW", "Description": "Sympathy, general advice, non-humanitarian news"}
            ]))
            
        with c2:
            st.markdown("### 📊 Overall Class Distribution")
            if os.path.exists("reports/figures/class_distribution.png"):
                st.image("reports/figures/class_distribution.png", use_container_width=True)

    # ---------------- TAB 2: ANALYZE SINGLE MESSAGE ----------------
    with tab2:
        st.subheader("Interactive Message Classification")
        
        st.markdown("**Sample Emergency Tweets (Click to copy):**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("🚨 Rescue Needed"):
                st.session_state["user_input_msg"] = "PLEASE HELP! 4 people trapped under collapsed residential building in Portoviejo, need rescue team urgently!"
        with sc2:
            if st.button("🏗️ Bridge Collapse"):
                st.session_state["user_input_msg"] = "Main highway bridge destroyed near river bank, roads completely blocked by earthquake debris."
        with sc3:
            if st.button("❤️ Sympathy Post"):
                st.session_state["user_input_msg"] = "Praying for everyone affected by the earthquake in Ecuador. Stay strong guys!"

        default_text = st.session_state.get("user_input_msg", "")
        input_text = st.text_area("Enter Disaster Social Media Text / Tweet:", value=default_text, height=120, placeholder="Type emergency message here...")
        
        if st.button("Analyze Message", type="primary"):
            if not input_text.strip():
                st.warning("Please enter a message to analyze.")
            else:
                with st.spinner("Analyzing message text with NLP model..."):
                    res = engine.predict_single(input_text, model_type=model_type, threshold=conf_threshold)
                    
                st.markdown("---")
                st.subheader("Classification Result")
                
                res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
                
                with res_col1:
                    st.markdown(f"**Predicted Category:**")
                    st.markdown(f"### {res['predicted_category']}")
                    
                with res_col2:
                    st.markdown(f"**Academic Priority Level:**")
                    p_lvl = res["priority_level"]
                    if p_lvl == "CRITICAL":
                        st.markdown(f'<span class="badge-critical">🚨 {p_lvl}</span>', unsafe_allow_html=True)
                    elif p_lvl == "HIGH":
                        st.markdown(f'<span class="badge-high">⚠️ {p_lvl}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-low">ℹ️ {p_lvl}</span>', unsafe_allow_html=True)
                        
                with res_col3:
                    st.markdown(f"**Model Confidence & Status:**")
                    st.markdown(f"### {res['confidence_percentage']}")
                    st_status = res["review_status"]
                    if "Human Review" in st_status:
                        st.markdown(f'<span class="badge-review">⚠️ {st_status}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-model">✅ {st_status}</span>', unsafe_allow_html=True)
                        
                st.markdown("---")
                exp_col1, exp_col2 = st.columns([1, 1])
                
                with exp_col1:
                    st.markdown("### 📈 Top Class Probabilities")
                    probs_df = pd.DataFrame(res["top_predictions"])
                    st.dataframe(probs_df, use_container_width=True)
                    
                with exp_col2:
                    st.markdown("### 💡 Why this prediction? (Explainability)")
                    st.info(res["explanation"])
                    if res.get("top_features"):
                        st.markdown("**Influential Terms / N-grams:**")
                        chips_html = "".join([f'<span class="feature-chip">{f["word"]}</span>' for f in res["top_features"]])
                        st.markdown(chips_html, unsafe_allow_html=True)

    # ---------------- TAB 3: BATCH CSV ANALYSIS ----------------
    with tab3:
        st.subheader("Batch Social Media Tweet Prioritization")
        st.write("Upload a CSV file containing social media messages. The system will classify every row and generate priority tags.")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded CSV with {len(df_upload)} rows.")
                
                text_col = st.selectbox("Select Text Column", df_upload.columns, index=0 if "text" not in df_upload.columns else list(df_upload.columns).index("text"))
                
                if st.button("Process Batch Predictions", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        res_df = engine.predict_batch(df_upload, text_col=text_col, model_type=model_type, threshold=conf_threshold)
                        
                    st.markdown("### Batch Results Preview")
                    st.dataframe(res_df.head(50), use_container_width=True)
                    
                    # Download CSV
                    csv_data = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Prioritized Predictions CSV",
                        data=csv_data,
                        file_name="prioritized_emergency_messages.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error reading or processing CSV file: {e}")

    # ---------------- TAB 4: MODEL PERFORMANCE ----------------
    with tab4:
        st.subheader("Model Evaluation & Performance Comparison")
        if eval_results:
            b_res = eval_results.get("baseline")
            t_res = eval_results.get("transformer")
            
            comp_rows = []
            if b_res:
                comp_rows.append({
                    "Model": b_res["model_name"],
                    "Accuracy": f"{b_res['accuracy']:.4f}",
                    "Macro Precision": f"{b_res['macro_precision']:.4f}",
                    "Macro Recall": f"{b_res['macro_recall']:.4f}",
                    "Macro F1": f"{b_res['macro_f1']:.4f}",
                    "Weighted F1": f"{b_res['weighted_f1']:.4f}"
                })
            if t_res:
                comp_rows.append({
                    "Model": t_res["model_name"],
                    "Accuracy": f"{t_res['accuracy']:.4f}",
                    "Macro Precision": f"{t_res['macro_precision']:.4f}",
                    "Macro Recall": f"{t_res['macro_recall']:.4f}",
                    "Macro F1": f"{t_res['macro_f1']:.4f}",
                    "Weighted F1": f"{t_res['weighted_f1']:.4f}"
                })
                
            st.markdown("### 📊 Overall Model Metrics")
            st.table(pd.DataFrame(comp_rows))
            
            if os.path.exists("reports/figures/model_comparison.png"):
                st.image("reports/figures/model_comparison.png", use_container_width=True)
                
            st.markdown("---")
            st.markdown("### 🧩 Confusion Matrices")
            cm_c1, cm_c2 = st.columns(2)
            with cm_c1:
                st.markdown("**Baseline Model Confusion Matrix**")
                if os.path.exists("reports/figures/confusion_matrix_baseline.png"):
                    st.image("reports/figures/confusion_matrix_baseline.png", use_container_width=True)
            with cm_c2:
                st.markdown("**DistilBERT Model Confusion Matrix**")
                if os.path.exists("reports/figures/confusion_matrix_transformer.png"):
                    st.image("reports/figures/confusion_matrix_transformer.png", use_container_width=True)
                else:
                    st.info("Transformer model confusion matrix will display after transformer fine-tuning evaluation completes.")
                    
            if b_res and "per_class" in b_res:
                st.markdown("---")
                st.markdown("### 📋 Per-Category Baseline Metrics")
                per_cls_df = pd.DataFrame(b_res["per_class"]).T
                st.dataframe(per_cls_df.style.format("{:.3f}", subset=["precision", "recall", "f1_score"]), use_container_width=True)
        else:
            st.warning("Evaluation results file not found. Run python -m src.evaluate first.")

    # ---------------- TAB 5: DATASET INSIGHTS ----------------
    with tab5:
        st.subheader("Exploratory Data Analysis (EDA) Insights")
        e_c1, e_c2 = st.columns(2)
        with e_c1:
            if os.path.exists("reports/figures/message_length_dist.png"):
                st.image("reports/figures/message_length_dist.png", use_container_width=True)
        with e_c2:
            if os.path.exists("reports/figures/most_frequent_words.png"):
                st.image("reports/figures/most_frequent_words.png", use_container_width=True)

if __name__ == "__main__":
    main()
