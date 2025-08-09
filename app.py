import sys
print("Python executable:", sys.executable)
print("sys.path:", sys.path)

import os
import math
import random
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from fer import FER
from PIL import Image
import cv2
import tempfile
import sys
import moviepy.editor



# --------------------------
# Config and file names
# --------------------------
SYNTH_ROWS = 2000
PREP_MODEL_PKL = "prep_time_model.pkl"
SAT_MODEL_PKL = "satisfaction_model.pkl"
SYNTH_PREP_CSV = "synth_prep_dataset.csv"
SYNTH_SAT_CSV = "synth_satisfaction_dataset.csv"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --------------------------
# Generate synthetic data function
# --------------------------
def generate_synth_data(n=SYNTH_ROWS):
    rows_prep = []
    rows_sat = []
    for i in range(n):
        steep_time = np.random.randint(60, 301)
        water_temp = np.random.randint(70, 101)
        milk_temp = 0 if np.random.rand() < 0.3 else np.random.randint(4, 41)
        sugar_level = round(np.random.uniform(0, 15), 1)
        tea_type = np.random.choice([1,2,3])
        queue_length = np.random.randint(0, 11)
        prep_minutes = 0.5 + (steep_time / 60.0)*0.7 + queue_length * 1.2 + np.random.normal(0,0.5)
        prep_minutes = max(0.5, prep_minutes)
        rows_prep.append({
            "steep_time": steep_time,
            "water_temp": water_temp,
            "milk_temp": milk_temp,
            "sugar_level": sugar_level,
            "tea_type_code": tea_type,
            "queue_length": queue_length,
            "prep_time_min": prep_minutes
        })
        if tea_type == 1:
            opt_temp, opt_steep = 95, 180
        elif tea_type == 2:
            opt_temp, opt_steep = 85, 120
        else:
            opt_temp, opt_steep = 80, 90
        temp_pen = ((water_temp - opt_temp)/8.0)**2
        steep_pen = ((steep_time - opt_steep)/60.0)**2
        milk_pen = 0 if milk_temp==0 else ((milk_temp-5)/20.0)**2
        sugar_pen = (sugar_level/10.0)**1.3
        sat_raw = 10.0 -  (temp_pen*1.5 + steep_pen*1.0 + milk_pen*1.0 + sugar_pen*0.8)
        shop_pen = (queue_length/15.0)
        sat = sat_raw - 0.5*shop_pen + np.random.normal(0,0.5)
        sat = max(2.5, min(10.0, sat))
        rows_sat.append({
            "steep_time": steep_time,
            "water_temp": water_temp,
            "milk_temp": milk_temp,
            "sugar_level": sugar_level,
            "tea_type_code": tea_type,
            "satisfaction": round(sat, 2)
        })
    return pd.DataFrame(rows_prep), pd.DataFrame(rows_sat)

# --------------------------
# Load or generate datasets
# --------------------------
if not (os.path.exists(SYNTH_PREP_CSV) and os.path.exists(SYNTH_SAT_CSV)):
    df_prep, df_sat = generate_synth_data(SYNTH_ROWS)
    df_prep.to_csv(SYNTH_PREP_CSV, index=False)
    df_sat.to_csv(SYNTH_SAT_CSV, index=False)
else:
    df_prep = pd.read_csv(SYNTH_PREP_CSV)
    df_sat = pd.read_csv(SYNTH_SAT_CSV)

# --------------------------
# Train or load models
# --------------------------
def train_prep_model(df):
    X = df[['steep_time','water_temp','milk_temp','sugar_level','tea_type_code','queue_length']]
    y = df['prep_time_min']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = RandomForestRegressor(n_estimators=150, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    joblib.dump(model, PREP_MODEL_PKL)
    return model, rmse

def train_sat_model(df):
    X = df[['steep_time','water_temp','milk_temp','sugar_level','tea_type_code']]
    y = df['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = RandomForestRegressor(n_estimators=150, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    joblib.dump(model, SAT_MODEL_PKL)
    return model, rmse

if os.path.exists(PREP_MODEL_PKL):
    prep_model = joblib.load(PREP_MODEL_PKL)
else:
    prep_model, prep_rmse = train_prep_model(df_prep)

if os.path.exists(SAT_MODEL_PKL):
    sat_model = joblib.load(SAT_MODEL_PKL)
else:
    sat_model, sat_rmse = train_sat_model(df_sat)

# --------------------------
# Build graph with user-editable weights
# --------------------------
def create_graph(edge_weights):
    G = nx.DiGraph()
    G.add_edge("Home", "ShopA", weight=edge_weights["Home-ShopA"])
    G.add_edge("Home", "ShopB", weight=edge_weights["Home-ShopB"])
    G.add_edge("Home", "ShopC", weight=edge_weights["Home-ShopC"])
    G.add_edge("ShopA", "ShopB", weight=edge_weights["ShopA-ShopB"])
    G.add_edge("ShopB", "ShopC", weight=edge_weights["ShopB-ShopC"])
    G.add_edge("ShopA", "ShopC", weight=edge_weights["ShopA-ShopC"])
    return G

pos = {
    "Home": (0,0),
    "ShopA": (2,1.5),
    "ShopB": (3, -0.5),
    "ShopC": (5, 1.0)
}

def plot_interactive_graph(G, pos, path_nodes):
    edges = list(G.edges(data=True))

    fig = go.Figure()

    # Draw all edges
    for u,v,data in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = 'orange' if (len(path_nodes) > 1 and (u,v) in list(zip(path_nodes[:-1], path_nodes[1:]))) else 'gray'
        width = 4 if color == 'orange' else 1
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(color=color, width=width),
            hoverinfo='none',
            showlegend=False
        ))

    # Draw nodes
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        marker=dict(color='lightblue', size=25, line=dict(color='DarkSlateGrey', width=2)),
        hoverinfo='text',
        showlegend=False
    ))

    # Edge weights annotations
    for u,v,data in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data['weight']
        fig.add_annotation(
            x=(x0+x1)/2,
            y=(y0+y1)/2,
            text=f"{weight:.1f}m",
            showarrow=False,
            font=dict(color="black", size=12)
        )

    fig.update_layout(
        title="Travel Graph with Shortest Path Highlighted",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Coffee recommendation by emotion
# --------------------------
def recommend_coffee(emotion):
    # Map emotions to coffee types & messages
    recommendations = {
        'angry': ("Espresso", "Strong & bold to calm your nerves"),
        'disgust': ("Latte", "Smooth and creamy for a fresh start"),
        'fear': ("Mocha", "Comforting chocolatey delight"),
        'happy': ("Cappuccino", "Light and bubbly to keep the smile"),
        'sad': ("Hot Chocolate", "Warm and soothing for tough times"),
        'surprise': ("Irish Coffee", "Exciting and uplifting"),
        'neutral': ("Americano", "Simple and classic"),
    }
    return recommendations.get(emotion, ("Americano", "Simple and classic"))

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Tea-Time Optimizer & Mood Coffee ☕", layout="wide", page_icon="🍵")

# Custom CSS to change label and text colors to black
st.markdown(
    """
    <style>
    /* Change general text color in labels, headers, sidebar labels */
    .css-1d391kg, /* main top padding container */
    .css-1v3fvcr, /* sidebar labels */
    .css-1y4p8pa, /* widget labels */
    .css-1v0mbdj, /* markdown text */
    .st-bx,       /* container backgrounds (keep white bg) */
    .stButton>button {
        color: black !important;  /* force text to black */
    }

    /* Change slider text color (input number) */
    .stSlider > div > div > input { 
        color: black !important; 
        font-weight: 700;
    }

    /* Change button background color and text color */
    .stButton>button { 
        background-color: #d43f3a; 
        color: white; 
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍵 Algorithmic Tea-Time Optimizer & Mood Coffee Recommender")
st.markdown("Find the fastest & tastiest tea with smart routing and get a coffee recommendation based on your mood!")

# --- Webcam Emotion Detection ---
st.subheader("😊 Detect Your Mood with Webcam")

img_file_buffer = st.camera_input("Take a selfie for mood-based coffee recommendation")

detector = FER(mtcnn=True)

if img_file_buffer is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(img_file_buffer)
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect emotions
    emotions = detector.detect_emotions(img)

    if emotions:
        dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        st.success(f"Detected Mood: *{dominant_emotion.capitalize()}*")
        coffee, desc = recommend_coffee(dominant_emotion)
        st.markdown(f"### ☕ Mood-based Coffee Recommendation: *{coffee}*")
        st.write(desc)
    else:
        st.warning("Face not detected. Please try again.")

st.markdown("---")

# Sidebar inputs for travel times and settings
with st.sidebar:
    st.header("🛣 Travel Times (minutes)")
    edge_weights = {
        "Home-ShopA": st.number_input("Home → ShopA", min_value=0.1, max_value=60.0, value=6.0, step=0.1, help="Travel time from Home to Shop A"),
        "Home-ShopB": st.number_input("Home → ShopB", min_value=0.1, max_value=60.0, value=9.0, step=0.1, help="Travel time from Home to Shop B"),
        "Home-ShopC": st.number_input("Home → ShopC", min_value=0.1, max_value=60.0, value=12.0, step=0.1, help="Travel time from Home to Shop C"),
        "ShopA-ShopB": st.number_input("ShopA → ShopB", min_value=0.1, max_value=60.0, value=4.0, step=0.1, help="Travel time from Shop A to Shop B"),
        "ShopB-ShopC": st.number_input("ShopB → ShopC", min_value=0.1, max_value=60.0, value=3.0, step=0.1, help="Travel time from Shop B to Shop C"),
        "ShopA-ShopC": st.number_input("ShopA → ShopC", min_value=0.1, max_value=60.0, value=7.0, step=0.1, help="Travel time from Shop A to Shop C"),
    }

    traffic_delay_pct = st.slider("🚦 Traffic Delay (%)", min_value=0, max_value=100, value=0, step=5, help="Simulate increased travel times due to traffic")
    preference_score = st.slider("⚖ Preference: 0=Fastest Tea, 1=Best Taste", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Balance between time and taste satisfaction")

st.markdown("---")

# Apply traffic delay
for k in edge_weights:
    edge_weights[k] *= 1 + (traffic_delay_pct / 100.0)

G = create_graph(edge_weights)

tea_type_map = {1: "Assam", 2: "Darjeeling", 3: "Green"}

# Input tea parameters in two columns
col1, col2 = st.columns(2)

with col1:
    desired_tea_type = st.selectbox("Select Tea Type", options=[1, 2, 3], format_func=lambda x: tea_type_map[x], help="Choose your preferred tea variety")
    desired_steep = st.slider("Steep Time (seconds)", min_value=60, max_value=300, value=180, help="Duration to steep your tea")
    desired_water = st.slider("Water Temperature (°C)", min_value=70, max_value=100, value=95, help="Ideal temperature for water")

with col2:
    desired_milk = st.slider("Milk Temperature (°C, 0 for no milk)", min_value=0, max_value=40, value=5, help="Temperature of milk to add or 0 for none")
    desired_sugar = st.slider("Sugar Level (g)", min_value=0.0, max_value=15.0, value=5.0, help="Amount of sugar added")

st.markdown("---")

st.subheader("⏳ Queue Lengths at Shops (randomized)")
shop_queues = {
    "ShopA": random.randint(0,7),
    "ShopB": random.randint(0,7),
    "ShopC": random.randint(0,7)
}
queue_cols = st.columns(3)
for i, shop in enumerate(["ShopA", "ShopB", "ShopC"]):
    queue_cols[i].metric(f"{shop}", f"{shop_queues[shop]} people waiting", delta=None)

# Calculate options
options = {}
for shop in ["ShopA", "ShopB", "ShopC"]:
    try:
        travel_time = nx.shortest_path_length(G, source="Home", target=shop, weight="weight")
    except nx.NetworkXNoPath:
        travel_time = float('inf')
    X_shop = np.array([[desired_steep, desired_water, desired_milk, desired_sugar, desired_tea_type, shop_queues[shop]]])
    pred_prep = prep_model.predict(X_shop)[0]
    total_time = travel_time + pred_prep

    X_sat = np.array([[desired_steep, desired_water, desired_milk, desired_sugar, desired_tea_type]])
    pred_satisfaction = sat_model.predict(X_sat)[0]

    norm_time = min(total_time / 30.0, 1.0)
    score = (1 - preference_score) * (1 - norm_time) + preference_score * (pred_satisfaction / 10.0)

    options[shop] = {
        "travel": travel_time,
        "prep": pred_prep,
        "total": total_time,
        "satisfaction": pred_satisfaction,
        "score": score
    }

# Home brew option
X_home = np.array([[desired_steep, desired_water, desired_milk, desired_sugar, desired_tea_type, 0]])
pred_prep_home = prep_model.predict(X_home)[0]
total_home = pred_prep_home
pred_satisfaction_home = sat_model.predict(np.array([[desired_steep, desired_water, desired_milk, desired_sugar, desired_tea_type]]))[0]
norm_time_home = min(total_home / 30.0, 1.0)
score_home = (1 - preference_score) * (1 - norm_time_home) + preference_score * (pred_satisfaction_home / 10.0)
options["Home"] = {
    "travel": 0.0,
    "prep": pred_prep_home,
    "total": total_home,
    "satisfaction": pred_satisfaction_home,
    "score": score_home
}

best_choice = max(options.items(), key=lambda kv: kv[1]['score'])
choice_name, choice_info = best_choice

# Show options as a table with metrics
st.subheader("☕ Tea Shop Options")
option_data = []
for shop, vals in options.items():
    option_data.append({
        "Shop": shop,
        "Travel Time (min)": round(vals["travel"], 2),
        "Prep Time (min)": round(vals["prep"], 2),
        "Total Time (min)": round(vals["total"], 2),
        "Satisfaction": round(vals["satisfaction"], 2),
        "Score": round(vals["score"], 3)
    })
df_options = pd.DataFrame(option_data)
st.dataframe(df_options.style.highlight_max(subset=["Score"], color="lightgreen"), height=220)

st.markdown(f"### 🏆 Best Choice: *{choice_name}* with score {choice_info['score']:.3f}")

# Show graph with shortest path highlighted
path_nodes = []
if choice_name != "Home":
    try:
        path_nodes = nx.shortest_path(G, source="Home", target=choice_name, weight="weight")
    except nx.NetworkXNoPath:
        path_nodes = []
else:
    path_nodes = ["Home"]

plot_interactive_graph(G, pos, path_nodes)