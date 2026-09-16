import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Smart Irrigation Prediction",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Smart Irrigation Prediction")

st.write(
    "AI-Based Crop Water Requirement and Irrigation Prediction System"
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

DATA_FILE = "irrigation_dataset.csv"

if not os.path.exists(DATA_FILE):
    st.error("irrigation_dataset.csv file not found.")
    st.stop()

df = pd.read_csv(DATA_FILE)


# ============================================================
# LABEL ENCODING
# ============================================================

le_crop = LabelEncoder()
le_stage = LabelEncoder()

df["Crop_Type_Enc"] = le_crop.fit_transform(
    df["Crop_Type"]
)

df["Growth_Stage_Enc"] = le_stage.fit_transform(
    df["Growth_Stage"]
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "Temperature",
    "Humidity",
    "Soil_Moisture",
    "Rainfall",
    "Crop_Type_Enc",
    "Growth_Stage_Enc"
]

X = df[features]

y_reg = df["Water_Requirement_mm"]

y_clf = df["Irrigation_Needed"]


# ============================================================
# TRAIN REGRESSION MODEL
# ============================================================

@st.cache_resource
def train_regression_model():

    model = XGBRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y_reg)

    return model


# ============================================================
# TRAIN CLASSIFICATION MODEL
# ============================================================

@st.cache_resource
def train_classification_model():

    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X, y_clf)

    return model


reg_model = train_regression_model()

clf_model = train_classification_model()


# ============================================================
# ENVIRONMENTAL PARAMETERS
# ============================================================

st.subheader("Environmental Parameters")


temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=60.0,
    step=0.1,
    value=None,
    placeholder="Enter temperature"
)


humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    value=None,
    placeholder="Enter humidity"
)


soil_moisture = st.number_input(
    "Soil Moisture (%)",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    value=None,
    placeholder="Enter soil moisture"
)


rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    value=None,
    placeholder="Enter rainfall"
)


# ============================================================
# CROP INFORMATION
# ============================================================

st.subheader("Crop Information")


crop = st.selectbox(
    "Crop Type",
    options=le_crop.classes_,
    index=None,
    placeholder="Select crop type"
)


stage = st.selectbox(
    "Growth Stage",
    options=le_stage.classes_,
    index=None,
    placeholder="Select growth stage"
)


st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔮 Predict",
    use_container_width=True
):

    # ========================================================
    # CHECK EMPTY INPUTS
    # ========================================================

    if (
        temperature is None
        or humidity is None
        or soil_moisture is None
        or rainfall is None
        or crop is None
        or stage is None
    ):

        st.warning(
            "⚠️ Please enter all environmental parameters "
            "and select the crop and growth stage."
        )

    else:

        try:

            # =================================================
            # ENCODE CROP
            # =================================================

            crop_enc = le_crop.transform(
                [crop]
            )[0]


            # =================================================
            # ENCODE GROWTH STAGE
            # =================================================

            stage_enc = le_stage.transform(
                [stage]
            )[0]


            # =================================================
            # CREATE INPUT DATA
            # =================================================

            input_data = pd.DataFrame(
                [[
                    temperature,
                    humidity,
                    soil_moisture,
                    rainfall,
                    crop_enc,
                    stage_enc
                ]],
                columns=features
            )


            # =================================================
            # WATER REQUIREMENT PREDICTION
            # =================================================

            water_req = reg_model.predict(
                input_data
            )[0]


            # =================================================
            # IRRIGATION PREDICTION
            # =================================================

            irrigation_flag = clf_model.predict(
                input_data
            )[0]


            # Prevent negative value
            water_req = max(
                0,
                water_req
            )


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.subheader("Prediction Result")


            if irrigation_flag == 1:

                st.success(
                    "Prediction: YES - Irrigation Needed"
                )

            else:

                st.success(
                    "Prediction: NO - Irrigation Not Needed"
                )


            st.info(
                f"Estimated Water Requirement: "
                f"{water_req:.2f} mm"
            )


        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Smart Irrigation and Crop Water Requirement "
    "Prediction System"
)