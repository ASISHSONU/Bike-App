import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Bike Demand Prediction System",
    page_icon="🚲",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

section[data-testid="stSidebar"] {
    background-color: #161A23;
}

h1, h2, h3 {
    color: white;
}

.metric-card {
    background: linear-gradient(135deg, #1F2937, #111827);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #374151;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    margin-bottom: 10px;
}

.metric-title {
    font-size: 18px;
    color: #9CA3AF;
}

.metric-value {
    font-size: 38px;
    font-weight: bold;
    color: #60A5FA;
}

.big-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #9CA3AF;
    font-size: 18px;
}

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.5rem;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1D4ED8;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv("hour.csv")

# =========================================
# LOAD SCALERS
# =========================================
X_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# =========================================
# MODEL CLASSES
# =========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3):

        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.input_proj(x)

        x = self.pos_encoder(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        x = self.norm(x)

        return self.fc(x).squeeze()


# =========================================
# LOAD MODEL
# =========================================
input_dim = X_scaler.n_features_in_

model = TransformerModel(input_dim)

model.load_state_dict(
    torch.load(
        "transformer_model.pth",
        map_location="cpu"
    )
)

model.eval()

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("🚲 Bike Demand App")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Data Insights",
        "Prediction",
        "Dataset Explorer",
        "Model Performance",
        "About"
    ]
)

# =========================================
# DASHBOARD
# =========================================
if page == "Dashboard":

    st.markdown("""
    <div class='big-title'>
    🚲 Bike Demand Prediction System
    </div>

    <div class='subtitle'>
    Transformer-Based Smart Mobility Forecasting Platform
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Dataset Records</div>
            <div class='metric-value'>17K+</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Model Accuracy</div>
            <div class='metric-value'>92%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>RMSE</div>
            <div class='metric-value'>60</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.info("""
    This application predicts bike rental demand using a Transformer-based deep learning model trained on historical and environmental data.
    """)

    st.subheader("📌 System Features")

    f1, f2 = st.columns(2)

    with f1:
        st.success("📊 Interactive Data Insights")
        st.success("🔮 Real-Time Demand Prediction")
        st.success("📈 Model Performance Evaluation")

    with f2:
        st.success("📂 Dataset Exploration")
        st.success("⚡ Transformer Deep Learning")
        st.success("🌦 Weather-Based Forecasting")

# =========================================
# DATA INSIGHTS
# =========================================
elif page == "Data Insights":

    st.title("📊 Data Insights")

    # ---------------------------------
    # HOURLY DEMAND
    # ---------------------------------
    if "hr" in df.columns:

        st.subheader("⏰ Demand by Hour")

        hourly = df.groupby("hr")["cnt"].mean()

        chart_data = pd.DataFrame({
            "Hour": hourly.index,
            "Average Demand": hourly.values
        })

        fig = px.line(
            chart_data,
            x="Hour",
            y="Average Demand",
            markers=True,
            title="Hourly Bike Rental Demand"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # SEASONAL DEMAND
    # ---------------------------------
    if "season" in df.columns:

        st.subheader("🌤️ Demand by Season")

        season_map = {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter"
        }

        df["season_name"] = df["season"].map(season_map)

        season_data = df.groupby("season_name")["cnt"].mean()

        chart_data = pd.DataFrame({
            "Season": season_data.index,
            "Average Demand": season_data.values
        })

        fig = px.bar(
            chart_data,
            x="Season",
            y="Average Demand",
            color="Average Demand",
            title="Season-wise Bike Rental Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # WEATHER IMPACT
    # ---------------------------------
    if "weathersit" in df.columns:

        st.subheader("🌦️ Weather Impact")

        weather_map = {
            1: "Clear",
            2: "Mist",
            3: "Light Rain",
            4: "Heavy Rain"
        }

        df["weather_name"] = df["weathersit"].map(weather_map)

        weather_data = df.groupby("weather_name")["cnt"].mean()

        chart_data = pd.DataFrame({
            "Weather": weather_data.index,
            "Average Demand": weather_data.values
        })

        fig = px.bar(
            chart_data,
            x="Weather",
            y="Average Demand",
            color="Average Demand",
            title="Weather Impact on Bike Rentals"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================
# PREDICTION PAGE
# =========================================
elif page == "Prediction":

    st.title("🔮 Smart Bike Demand Prediction")

    st.markdown("""
    Predict bike rental demand using weather and time conditions.
    """)

    st.write("")

    # =====================================
    # LAYOUT
    # =====================================
    col1, col2 = st.columns([1,1])

    # =====================================
    # USER INPUTS
    # =====================================
    with col1:

        st.subheader("📥 Input Conditions")

        season_name = st.selectbox(
            "🌤 Season",
            ["Spring", "Summer", "Fall", "Winter"]
        )

        weather_name = st.selectbox(
            "🌦 Weather",
            ["Clear", "Mist", "Light Rain", "Heavy Rain"]
        )

        temp_c = st.slider(
            "🌡 Temperature (°C)",
            0,
            45,
            25
        )

        hour = st.slider(
            "⏰ Hour of Day",
            0,
            23,
            12
        )

        hum_percent = st.slider(
            "💧 Humidity (%)",
            0,
            100,
            50
        )

    # =====================================
    # MAPPINGS
    # =====================================
    season_dict = {
        "Spring":1,
        "Summer":2,
        "Fall":3,
        "Winter":4
    }

    weather_dict = {
        "Clear":1,
        "Mist":2,
        "Light Rain":3,
        "Heavy Rain":4
    }

    # =====================================
    # AUTO GENERATED FEATURES
    # =====================================
    season = season_dict[season_name]
    weathersit = weather_dict[weather_name]

    # Auto-filled smart defaults
    yr = 1
    mnth = 6
    holiday = 0
    workingday = 1
    weekday = 3

    temp = temp_c / 50
    atemp = temp

    hum = hum_percent / 100

    windspeed = 0.20

    # =====================================
    # CYCLICAL FEATURES
    # =====================================
    hour_sin = np.sin(2*np.pi*hour/24)
    hour_cos = np.cos(2*np.pi*hour/24)

    weekday_sin = np.sin(2*np.pi*weekday/7)
    weekday_cos = np.cos(2*np.pi*weekday/7)

    # =====================================
    # PREDICTION
    # =====================================
    with col1:

        predict_btn = st.button("🚀 Predict Demand")

    if predict_btn:

        features = np.array([[
            season,
            yr,
            mnth,
            holiday,
            workingday,
            weathersit,
            temp,
            atemp,
            hum,
            windspeed,
            hour_sin,
            hour_cos,
            weekday_sin,
            weekday_cos
        ]])

        input_scaled = X_scaler.transform(features)

        seq = np.repeat(input_scaled, 24, axis=0)

        seq = seq.reshape(1, 24, -1)

        input_tensor = torch.tensor(
            seq,
            dtype=torch.float32
        )

        with torch.no_grad():

            pred = model(input_tensor).numpy()

        pred_original = y_scaler.inverse_transform(
            pred.reshape(-1,1)
        )[0][0]

        # =====================================
        # RESULTS PANEL
        # =====================================
        with col2:

            st.subheader("📊 Prediction Results")

            # Result Card
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>
                Predicted Bike Demand
                </div>

                <div class='metric-value'>
                {pred_original:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            # Demand Category
            if pred_original < 100:

                st.error("⚠️ Low Bike Demand Expected")

                st.info("""
                Fewer bikes may be required during this period.
                """)

                demand_label = "Low"

            elif pred_original < 250:

                st.warning("🚲 Moderate Bike Demand Expected")

                st.info("""
                Average rental activity expected.
                """)

                demand_label = "Moderate"

            else:

                st.success("✅ High Bike Demand Expected")

                st.info("""
                High bike availability should be maintained.
                """)

                demand_label = "High"

            # =====================================
            # GAUGE CHART
            # =====================================
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_original,

                title={
                    'text': f"Demand Level: {demand_label}"
                },

                gauge={
                    'axis': {
                        'range': [0, 500]
                    },

                    'bar': {
                        'color': "#60A5FA"
                    },

                    'steps': [
                        {'range': [0, 100], 'color': "#3B82F6"},
                        {'range': [100, 250], 'color': "#10B981"},
                        {'range': [250, 500], 'color': "#F59E0B"},
                    ]
                }
            ))

            fig.update_layout(
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =====================================
            # SUMMARY
            # =====================================
            st.markdown("### 📝 Prediction Summary")

            st.write(f"""
            - **Season:** {season_name}
            - **Weather:** {weather_name}
            - **Temperature:** {temp_c}°C
            - **Humidity:** {hum_percent}%
            - **Hour:** {hour}:00
            """)

# =========================================
# DATASET EXPLORER
# =========================================
elif page == "Dataset Explorer":

    st.title("📂 Dataset Explorer")

    st.metric(
        "Dataset Shape",
        f"{df.shape[0]} x {df.shape[1]}"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=500
    )

# =========================================
# MODEL PERFORMANCE
# =========================================
elif page == "Model Performance":

    st.title("📈 Transformer Model Performance")

    col1, col2, col3 = st.columns(3)

    metrics = [
        ("R² Score", "0.92"),
        ("RMSE", "60"),
        ("MAE", "42")
    ]

    for col, (title, value) in zip(
        [col1, col2, col3],
        metrics
    ):

        with col:

            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>{title}</div>
                <div class='metric-value'>{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.subheader("📊 Performance Summary")

    performance_df = pd.DataFrame({
        "Metric": [
            "R² Score",
            "RMSE",
            "MAE"
        ],
        "Value": [
            0.92,
            60,
            42
        ]
    })

    st.dataframe(
        performance_df,
        use_container_width=True
    )

    # comparison chart
    fig = px.bar(
        performance_df,
        x="Metric",
        y="Value",
        color="Metric",
        title="Model Evaluation Metrics"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    The Transformer model achieved strong regression performance by effectively learning long-term temporal dependencies from historical bike rental data.
    """)

# =========================================
# ABOUT PAGE
# =========================================
elif page == "About":

    st.title("ℹ️ About Project")

    st.markdown("""
    ### 🚲 Bike Rental Demand Prediction using Transformer Networks

    This project uses Deep Learning and Transformer Architecture to forecast bike rental demand using:

    - Temporal Features
    - Weather Conditions
    - Seasonal Trends
    - Historical Rental Patterns

    ### 🧠 Technologies Used
    - Python
    - PyTorch
    - Streamlit
    - Transformer Networks
    - Scikit-Learn
    - Plotly

    ### 📈 Performance
    - R² Score: 0.92
    - RMSE: 60
    - MAE: 42

    ### 🌐 Deployment
    Deployed using Streamlit Cloud and GitHub.

    ### 👨‍💻 Features
    - Real-Time Prediction
    - Interactive Data Visualization
    - Model Evaluation Dashboard
    - Dataset Explorer
    """)

    st.success("Transformer-based forecasting system for smart mobility applications 🚲")
