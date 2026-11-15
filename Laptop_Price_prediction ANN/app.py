
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path
import textwrap
# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LAPTOPIQ | Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at 10% 5%, #30205c 0%, transparent 30%),
        radial-gradient(circle at 90% 15%, #123e59 0%, transparent 30%),
        linear-gradient(135deg, #080c19, #10172d, #080c19);

    color: #ffffff;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* HERO SECTION */

.hero {
    padding: 40px;
    border-radius: 25px;
    background: linear-gradient(
        120deg,
        rgba(91, 33, 182, 0.45),
        rgba(30, 64, 175, 0.35),
        rgba(8, 145, 178, 0.25)
    );

    border: 1px solid rgba(167, 139, 250, 0.3);
    margin-bottom: 30px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
}

.hero-tag {
    color: #67e8f9;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 52px;
    font-weight: 700;
    color: white;
    margin: 12px 0;
}

.hero h1 span {
    background: linear-gradient(90deg, #c4b5fd, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

/* SECTION HEADINGS */

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #e0e7ff;
    margin: 20px 0;
}

/* INPUT LABELS */

.stSelectbox label,
.stSlider label,
.stNumberInput label {
    color: #dbeafe !important;
    font-weight: 600 !important;
}

/* INPUT FIELDS */

div[data-baseweb="select"] > div {
    background-color: #151e35 !important;
    border: 1px solid #354365 !important;
    border-radius: 12px !important;
    color: white !important;
}

div[data-baseweb="select"] span {
    color: white !important;
}

.stNumberInput input {
    background: #151e35 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* BUTTON */

.stFormSubmitButton button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    width: 100%;
    transition: 0.3s;
}

.stFormSubmitButton button:hover {
    box-shadow: 0 0 25px rgba(34, 211, 238, 0.35);
    transform: translateY(-2px);
}

/* PREDICTION CARD */

.result-card {
    background: linear-gradient(
        120deg,
        rgba(91, 33, 182, 0.35),
        rgba(8, 145, 178, 0.25)
    );

    border: 1px solid rgba(103, 232, 249, 0.35);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.3);
}

.result-label {
    color: #a5b4fc;
    font-size: 14px;
    letter-spacing: 2px;
    font-weight: 700;
}

.result-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 48px;
    font-weight: 700;
    color: #67e8f9;
    margin: 15px 0;
}

.result-note {
    color: #cbd5e1;
    font-size: 13px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 13px;
}

hr {
    border-color: rgba(148, 163, 184, 0.15);
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL AND PICKLE FILES
# ==========================================

@st.cache_resource
def load_assets():

    with open(BASE_DIR / "dropdowns.pkl", "rb") as f:
        dropdowns = pickle.load(f)

    with open(BASE_DIR / "model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = tf.keras.models.load_model(
        BASE_DIR / "laptop_price_model.keras",
        compile=False
    )

    return dropdowns, model_columns, scaler, model


try:
    dropdowns, model_columns, scaler, model = load_assets()

except Exception as e:
    st.error("Unable to load model files. Check your project folder.")
    st.exception(e)
    st.stop()

# ==========================================
# 4. HERO SECTION
# ==========================================

st.html("""
<div class="hero">

    <div class="hero-tag">
        ✦ AI POWERED PRICE INTELLIGENCE
    </div>

    <h1>
        LAPTOP<span>IQ</span>
    </h1>

    <h2 style="color:#e0e7ff; font-size:25px;">
        Discover the true value of your laptop.
    </h2>

    <p>
        An intelligent laptop price prediction system powered
        by Deep Learning and TensorFlow.
        Configure your laptop specifications and get an instant
        estimated price.
    </p>

</div>
""")

# ==========================================
# 5. LAPTOP SPECIFICATIONS FORM
# ==========================================

st.markdown(
    '<div class="section-heading">⚙️ Configure Your Laptop</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter your laptop specifications below to estimate its price."
)


def get_index(options, preferred):

    try:
        return options.index(preferred)
    except ValueError:
        return 0


with st.form("laptop_form"):

    col1, col2 = st.columns(2, gap="large")

    # --------------------------------------
    # LEFT COLUMN
    # --------------------------------------

    with col1:

        st.subheader("🏷️ Brand & Category")

        company = st.selectbox(
            "Laptop Brand",
            dropdowns["Company"],
            index=get_index(dropdowns["Company"], "Dell")
        )

        laptop_type = st.selectbox(
            "Laptop Type",
            dropdowns["TypeName"],
            index=get_index(dropdowns["TypeName"], "Notebook")
        )

        os_name = st.selectbox(
            "Operating System",
            dropdowns["OpSys"],
            index=get_index(dropdowns["OpSys"], "Windows 10")
        )

        st.divider()

        st.subheader("🖥️ Display & Design")

        screen = st.selectbox(
            "Screen Resolution",
            dropdowns["ScreenResolution"],
            index=get_index(
                dropdowns["ScreenResolution"],
                "1920x1080"
            )
        )

        inches = st.slider(
            "Screen Size (Inches)",
            min_value=10.0,
            max_value=18.0,
            value=15.6,
            step=0.1
        )

        weight = st.slider(
            "Laptop Weight (Kg)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        )

    # --------------------------------------
    # RIGHT COLUMN
    # --------------------------------------

    with col2:

        st.subheader("🚀 Performance")

        cpu = st.selectbox(
            "Processor",
            dropdowns["Cpu"]
        )

        gpu = st.selectbox(
            "Graphics Card",
            dropdowns["Gpu"]
        )

        ram = st.select_slider(
            "RAM (GB)",
            options=[2, 4, 6, 8, 12, 16, 24, 32, 64],
            value=8
        )

        st.divider()

        st.subheader("💾 Storage")

        memory = st.selectbox(
            "Storage Configuration",
            dropdowns["Memory"],
            index=get_index(
                dropdowns["Memory"],
                "256GB SSD"
            )
        )

    st.write("")

    submit = st.form_submit_button(
        "✦  PREDICT LAPTOP PRICE",
        use_container_width=True
    )


# ==========================================
# 6. PRICE PREDICTION
# ==========================================

if submit:

    with st.spinner("Analyzing laptop specifications..."):

        try:

            # Numerical features
            numeric_data = pd.DataFrame([{
                "Inches": float(inches),
                "Ram": int(ram),
                "Weight": float(weight)
            }])

            # Apply the saved scaler
            scaled_values = scaler.transform(
                numeric_data[["Inches", "Ram", "Weight"]]
            )[0]

            # Create input with the exact model columns
            input_data = pd.DataFrame(
                np.zeros((1, len(model_columns))),
                columns=model_columns
            )

            # Insert scaled numerical values
            numerical_features = [
                "Inches",
                "Ram",
                "Weight"
            ]

            for feature, value in zip(
                numerical_features,
                scaled_values
            ):

                if feature in input_data.columns:
                    input_data.loc[0, feature] = value

            # Categorical features
            categorical_data = {
                "Company": company,
                "TypeName": laptop_type,
                "ScreenResolution": screen,
                "Cpu": cpu,
                "Memory": memory,
                "Gpu": gpu,
                "OpSys": os_name
            }

            # One-hot encoding
            for category, value in categorical_data.items():

                column_name = f"{category}_{value}"

                if column_name in input_data.columns:
                    input_data.loc[0, column_name] = 1

            # Model prediction
            prediction = model.predict(
                input_data.to_numpy(dtype=np.float32),
                verbose=0
            )

            predicted_price = float(
                np.asarray(prediction).reshape(-1)[0]
            )

            if not np.isfinite(predicted_price):
                raise ValueError("Invalid model prediction.")

   
        
            # ==================================
            # 7. DISPLAY RESULTS
            # ==================================

            st.markdown("---")

            st.markdown(
                '<div class="section-heading">'
                '✨ Your Prediction Results'
                '</div>',
                unsafe_allow_html=True
            )

            # Premium prediction result card
            st.html(f"""
            <div class="result-card">

                <div class="result-label">
                    ESTIMATED LAPTOP PRICE
                </div>

                <div class="result-price">
                    € {predicted_price:,.2f}
                </div>

                <div class="result-note">
                    AI-powered estimate based on your laptop specifications
                </div>

            </div>
            """)

            # Configuration summary
            st.markdown("### 📋 Laptop Configuration Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Brand", company)
                st.metric("Processor", cpu)

            with summary_col2:
                st.metric("RAM", f"{ram} GB")
                st.metric("Storage", memory)

            with summary_col3:
                st.metric("Screen Size", f"{inches} Inches")
                st.metric("Weight", f"{weight} Kg")

            st.info(
                "Note: This is an estimated price, not a live market quote. "
                "Actual laptop prices may vary depending on condition, "
                "seller, region and availability."
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")



# ==========================================
# 8. FOOTER
# ==========================================

st.html("""
<div class="footer">

    <h3 style="color:#a5b4fc;">LAPTOPIQ</h3>

    <p>AI-Powered Laptop Price Prediction System</p>

    <p>Built with Python | Streamlit | TensorFlow</p>

    <p>© 2026 LAPTOPIQ. All Rights Reserved.</p>

</div>
""")
