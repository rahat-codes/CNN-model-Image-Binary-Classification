
from pathlib import Path
import streamlit as st

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_leaf_classifier.keras"
DATASET_PATH = PROJECT_ROOT / "dataset" / "split"

st.set_page_config(
    page_title="LeafVision | CNN Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main application */
.stApp {
    background: #0b111a;
    color: #e8edf5;
}

/* Hide Streamlit's automatic navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111c29;
    border-right: 1px solid #263748;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* Sidebar brand */
.sidebar-brand {
    padding: 12px 8px 25px 8px;
    border-bottom: 1px solid #293b4e;
    margin-bottom: 20px;
}

.sidebar-brand h2 {
    color: #f1f5f9;
    font-size: 25px;
    margin-bottom: 5px;
}

.sidebar-brand p {
    color: #94a3b8;
    font-size: 13px;
}

/* Navigation links */
[data-testid="stSidebar"] a {
    border-radius: 9px;
    padding: 9px 12px;
    transition: all 0.25s ease;
}

[data-testid="stSidebar"] a:hover {
    background: #24364a;
    transform: translateX(4px);
}

/* Page heading */
.page-heading {
    font-size: 34px;
    font-weight: 750;
    color: #f1f5f9;
    margin-bottom: 5px;
}

.page-subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 28px;
}

/* Animated hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 35px;
    border-radius: 18px;
    background: linear-gradient(
        120deg,
        #142b35,
        #172b3d,
        #1c2941
    );
    border: 1px solid #2c4753;
    animation: fadeUp 0.8s ease-out;
}

.hero::before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -80px;
    top: -110px;
    border-radius: 50%;
    background: #36d399;
    opacity: 0.12;
    filter: blur(55px);
    animation: glow 5s ease-in-out infinite alternate;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero h1 {
    font-size: 35px;
    color: #f8fafc;
    margin-bottom: 12px;
}

.hero p {
    color: #b6c5d5;
    font-size: 16px;
    max-width: 650px;
    line-height: 1.7;
}

.hero-leaf {
    position: absolute;
    right: 55px;
    top: 20px;
    font-size: 105px;
    animation: floatLeaf 4s ease-in-out infinite;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    background: #34d399;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 12px #34d399;
    animation: pulse 2s infinite;
}

.status-text {
    color: #6ee7b7;
    font-size: 13px;
    font-weight: 600;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #172536;
    border: 1px solid #293d52;
    padding: 18px;
    border-radius: 13px;
    transition: all 0.25s ease;
    animation: fadeUp 0.8s ease-out;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #36d399;
    box-shadow: 0 8px 25px #00000030;
}

div[data-testid="stMetricLabel"] {
    color: #9fb2c8;
}

div[data-testid="stMetricValue"] {
    color: #f1f5f9;
}

/* Information cards */
.info-card {
    background: #172536;
    border: 1px solid #293d52;
    border-radius: 13px;
    padding: 20px;
    height: 100%;
    transition: all 0.25s ease;
}

.info-card:hover {
    transform: translateY(-4px);
    border-color: #36d399;
}

.info-card h3 {
    color: #dbeafe;
    font-size: 17px;
}

.info-card p {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}

/* Section heading */
.section-heading {
    color: #e2e8f0;
    font-size: 22px;
    font-weight: 650;
    margin-top: 35px;
    margin-bottom: 16px;
}

/* Animations */
@keyframes floatLeaf {
    0%, 100% {
        transform: translateY(0px) rotate(-5deg);
    }
    50% {
        transform: translateY(-13px) rotate(5deg);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.4;
    }
}

@keyframes glow {
    from {
        transform: scale(0.9);
    }
    to {
        transform: scale(1.2);
    }
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive animation */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <h2>🍃 LeafVision</h2>
        <p>Apple Leaf Classification System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### MENU")

    st.page_link(
        "dashboard.py",
        label="Overview",
        icon="🏠"
    )

    st.page_link(
        "pages/1_Dataset.py",
        label="Dataset",
        icon="📁"
    )

    st.page_link(
        "pages/2_Training.py",
        label="Training",
        icon="🧠"
    )

    st.page_link(
        "pages/3_Evaluation.py",
        label="Evaluation",
        icon="📊"
    )

    st.page_link(
        "pages/4_Prediction.py",
        label="Prediction",
        icon="🔍"
    )

    st.markdown("---")

    st.caption("CNN IMAGE CLASSIFICATION")
    st.caption("TensorFlow · Keras · Streamlit")


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

def count_images(folder):

    if not folder.exists():
        return 0

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return sum(
        1 for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )


train_count = count_images(DATASET_PATH / "train")
validation_count = count_images(DATASET_PATH / "validation")
test_count = count_images(DATASET_PATH / "test")

total_images = train_count + validation_count + test_count


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.markdown(
    '<div class="page-heading">Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Monitor your CNN image classification system.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

model_status = MODEL_PATH.exists()
dataset_status = DATASET_PATH.exists()

st.markdown(f"""
<div class="hero">

    <div class="hero-content">

        <div class="status-text">
            <span class="status-dot"></span>
            SYSTEM ONLINE
        </div>

        <h1>Welcome to LeafVision</h1>

        <p>
            An interactive deep learning application that uses
            Convolutional Neural Networks to classify apple leaf
            images into Healthy and Unhealthy categories.
        </p>

    </div>

    <div class="hero-leaf">🍃</div>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">System Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model Status",
        "Ready" if model_status else "Missing",
        delta="Available" if model_status else "Not found"
    )

with col2:
    st.metric(
        "Dataset Images",
        f"{total_images:,}",
        delta="Total images"
    )

with col3:
    st.metric(
        "Input Size",
        "128 × 128",
        delta="RGB"
    )

with col4:
    st.metric(
        "Classes",
        "2",
        delta="Binary classification"
    )


# --------------------------------------------------
# DATASET DISTRIBUTION
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Dataset Distribution</div>',
    unsafe_allow_html=True
)

if total_images > 0:

    chart_col, detail_col = st.columns([1.5, 1])

    with chart_col:

        st.bar_chart({
            "Images": {
                "Training": train_count,
                "Validation": validation_count,
                "Testing": test_count
            }
        })

    with detail_col:

        st.markdown(
            f"""
            <div class="info-card">
                <h3>📁 Dataset Summary</h3>
                <p>Training: <b>{train_count}</b> images</p>
                <p>Validation: <b>{validation_count}</b> images</p>
                <p>Testing: <b>{test_count}</b> images</p>
                <p>Total: <b>{total_images}</b> images</p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.warning(
        "No dataset images found. Check your dataset/split directory."
    )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Model Information</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 Architecture</h3>
        <p>Convolutional Neural Network (CNN)</p>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="info-card">
        <h3>🌱 Classification</h3>
        <p>Healthy Leaf / Unhealthy Leaf</p>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="info-card">
        <h3>⚙️ Framework</h3>
        <p>TensorFlow / Keras</p>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "LeafVision | Apple Leaf Binary Classification | "
    "Built with TensorFlow and Streamlit"
)
