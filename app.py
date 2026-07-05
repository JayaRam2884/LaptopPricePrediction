import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# Load Artifacts
# ==========================================
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("LaptopPriceModel.pkl")
    feature_names = joblib.load("FeatureNames.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

st.title("💻 Laptop Price Predictor")
st.write("Fill in the specs below to estimate the price in euros.")

# ==========================================
# Raw Inputs (mirrors the original dataset columns)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", [
        "Apple", "HP", "Acer", "Asus", "Dell", "Lenovo", "MSI",
        "Toshiba", "Samsung", "Razer", "Chuwi", "Google", "Fujitsu",
        "LG", "Huawei", "Vero", "Mediacom", "Xiaomi"
    ])
    type_name = st.selectbox("Type", [
        "Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible",
        "Workstation", "Netbook"
    ])
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
    # Weight is a plain typeable number field (no slider) so exact values
    # like 1.37 kg can be entered directly.
    weight = st.number_input(
        "Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.01, format="%.2f"
    )
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    ips = st.selectbox("IPS Panel", ["No", "Yes"])

with col2:
    inches = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)

    # Screen Resolution: single combined dropdown instead of separate
    # horizontal/vertical selects.
    resolution_options = {
        "1366x768 (HD)": (1366, 768),
        "1440x900 (WXGA+)": (1440, 900),
        "1600x900 (HD+)": (1600, 900),
        "1920x1080 (Full HD)": (1920, 1080),
        "1920x1200 (WUXGA)": (1920, 1200),
        "2560x1440 (QHD)": (2560, 1440),
        "2560x1600": (2560, 1600),
        "2880x1800 (Retina)": (2880, 1800),
        "3200x1800 (QHD+)": (3200, 1800),
        "3840x2160 (4K UHD)": (3840, 2160),
    }
    resolution_label = st.selectbox(
        "Screen Resolution", list(resolution_options.keys()), index=3
    )
    x_res, y_res = resolution_options[resolution_label]

    cpu_brand = st.selectbox("CPU Brand", ["Intel Core i7", "Intel Core i5", "Intel Core i3", "Other Intel", "AMD"])
    gpu_brand = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD", "ARM"])
    os = st.selectbox("Operating System", ["Windows", "Mac", "Linux", "Others"])

st.subheader("Storage")
s1, s2, s3, s4 = st.columns(4)
with s1:
    ssd = st.number_input("SSD (GB)", min_value=0, max_value=2048, value=256, step=128)
with s2:
    hdd = st.number_input("HDD (GB)", min_value=0, max_value=2048, value=0, step=128)
with s3:
    flash = st.number_input("Flash Storage (GB)", min_value=0, max_value=1024, value=0, step=64)
with s4:
    hybrid = st.number_input("Hybrid (GB)", min_value=0, max_value=2048, value=0, step=128)

# ==========================================
# Feature Engineering (mirrors the training notebook)
# ==========================================
def build_feature_row():
    total_storage = ssd + hdd + flash + hybrid
    ppi = np.sqrt(x_res**2 + y_res**2) / inches

    row = {
        "Company": company,
        "TypeName": type_name,
        "Ram_GB": ram,
        "Weight_Kg": weight,
        "SSD": ssd,
        "HDD": hdd,
        "Flash_Storage": flash,
        "Hybrid": hybrid,
        "Total_Storage": total_storage,
        "Touchscreen": 1 if touchscreen == "Yes" else 0,
        "IPS": 1 if ips == "Yes" else 0,
        "PPI": ppi,
        "CPU_Brand": cpu_brand,
        "GPU_Brand": gpu_brand,
        "OS": os,
    }
    df_row = pd.DataFrame([row])

    cat_cols = ["Company", "TypeName", "CPU_Brand", "GPU_Brand", "OS"]
    df_encoded = pd.get_dummies(df_row, columns=cat_cols)

    # Clean column names the same way the training script did
    df_encoded.columns = (
        df_encoded.columns
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("<", "", regex=False)
    )

    # Align to the exact training feature set; any dummy not present here
    # (e.g. a company/category not selected) is filled with 0
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_aligned

# ==========================================
# Predict
# ==========================================
if st.button("Predict Price", type="primary"):
    X_input = build_feature_row()
    prediction = model.predict(X_input)[0]
    st.success(f"### Estimated Price: €{prediction:,.2f}")

    with st.expander("See engineered feature row"):
        st.dataframe(X_input.T.rename(columns={0: "value"}))