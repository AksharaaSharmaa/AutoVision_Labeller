import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ---------------------- CONFIG & STYLING ----------------------

st.set_page_config(page_title="AutoVision Labeler", layout="wide", initial_sidebar_state="expanded")

# --- Define Professional Color Palette ---
PRIMARY_COLOR = "#1A1A1A"      # Deep Black
SECONDARY_COLOR = "#FFFFFF"    # Pure White
ACCENT_COLOR = "#2563EB"       # Professional Blue
ACCENT_HOVER = "#1E40AF"       # Darker Blue
TEXT_PRIMARY = "#0F172A"       # Slate 900
TEXT_SECONDARY = "#64748B"     # Slate 500
BORDER_COLOR = "#E2E8F0"       # Slate 200
BG_LIGHT = "#F8FAFC"           # Slate 50
SIDEBAR_ACCENT = "#EFF6FF"     # Light Blue
ACCENT_COLOR_BGR = (235, 99, 37)  # BGR format for OpenCV

# --- Inject Custom CSS for Ultra-Professional Design ---
st.markdown(f"""
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Base Styling */
    html, body, [class*="st-"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {SECONDARY_COLOR} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    
    .main {{
        background-color: {SECONDARY_COLOR} !important;
        padding-top: 2rem !important;
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Clean Header - No Background Box */
    .header-container {{
        text-align: center;
        padding: 1rem 0 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 3px solid {ACCENT_COLOR};
    }}
    
    .header-title {{
        color: {PRIMARY_COLOR} !important;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -1px;
    }}
    
    .header-subtitle {{
        color: {TEXT_SECONDARY} !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-top: 0.75rem !important;
        letter-spacing: 0.3px;
    }}
    
    /* Section Headers */
    h1, h2, h3 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 600 !important;
    }}
    
    h2 {{
        font-size: 1.75rem !important;
        border-bottom: 2px solid {BORDER_COLOR};
        padding-bottom: 0.75rem;
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
    }}
    
    h3 {{
        font-size: 1.25rem !important;
        color: {TEXT_PRIMARY} !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }}

    /* Sidebar Professional Styling */
    [data-testid="stSidebar"] {{
        background: {SECONDARY_COLOR} !important;
        border-right: 2px solid {BORDER_COLOR} !important;
        box-shadow: 2px 0 8px rgba(0,0,0,0.05);
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.5rem;
    }}
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {TEXT_PRIMARY} !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }}
    
    /* Sidebar Info Boxes with Color */
    [data-testid="stSidebar"] .stAlert {{
        background: linear-gradient(135deg, {SIDEBAR_ACCENT} 0%, {BG_LIGHT} 100%) !important;
        border-left: 4px solid {ACCENT_COLOR} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* Reduce Sidebar Spacing */
    [data-testid="stSidebar"] .element-container {{
        margin-bottom: 0.5rem !important;
    }}
    
    [data-testid="stSidebar"] hr {{
        margin: 1rem 0 !important;
        background: {BORDER_COLOR};
        height: 1px;
    }}
    
    /* Professional Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT_COLOR} 0%, {ACCENT_HOVER} 100%) !important;
        color: {SECONDARY_COLOR} !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, {ACCENT_HOVER} 0%, #1E3A8A 100%) !important;
        box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.3);
        transform: translateY(-2px);
    }}
    
    /* Download Button Special Styling */
    .stDownloadButton > button {{
        background: {SECONDARY_COLOR} !important;
        color: {PRIMARY_COLOR} !important;
        border: 2px solid {PRIMARY_COLOR} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stDownloadButton > button:hover {{
        background: {PRIMARY_COLOR} !important;
        color: {SECONDARY_COLOR} !important;
        border: 2px solid {PRIMARY_COLOR} !important;
        transform: translateY(-2px);
    }}
    
    /* File Uploader */
    [data-testid="stFileUploader"] {{
        border: 2px dashed {ACCENT_COLOR} !important;
        background-color: {BG_LIGHT} !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {ACCENT_HOVER} !important;
        background-color: {SECONDARY_COLOR} !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
    }}

    /* Professional Metrics Cards */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, {BG_LIGHT} 0%, {SECONDARY_COLOR} 100%);
        border: 1px solid {BORDER_COLOR};
        border-left: 4px solid {ACCENT_COLOR} !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }}
    
    [data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }}
    
    [data-testid="stMetric"] label {{
        font-weight: 600 !important;
        color: {TEXT_SECONDARY} !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: {ACCENT_COLOR} !important;
    }}

    /* Dataframe Styling */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}

    /* Info/Warning/Success Boxes */
    .stAlert {{
        border-radius: 12px !important;
        border: 1px solid {BORDER_COLOR} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    /* Horizontal Separator */
    hr {{
        background: linear-gradient(90deg, transparent, {BORDER_COLOR}, transparent);
        height: 2px;
        border: none;
        margin: 2.5rem 0;
    }}
    
    /* Image Captions */
    .stImage > div {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid {BORDER_COLOR};
        transition: all 0.3s ease;
    }}
    
    .stImage > div:hover {{
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }}
    
    /* Select boxes and inputs */
    .stMultiSelect {{
        border-radius: 8px;
    }}
    
    .stMultiSelect > div {{
        border: 2px solid {PRIMARY_COLOR} !important;
        border-radius: 8px !important;
    }}
    
    /* Improve spacing */
    .block-container {{
        padding-top: 3rem;
        padding-bottom: 3rem;
    }}

</style>
""", unsafe_allow_html=True)


# ---------------------- CLEAN PROFESSIONAL HEADER ----------------------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">AutoVision Labeler</h1>
    <p class="header-subtitle">AI-Powered Image Annotation • YOLOv8 Technology • Professional Workflow</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
CLASS_NAMES = list(model.names.values())

# ---------------------- SIDEBAR DETECTION SETTINGS ----------------------
st.sidebar.markdown("### Detection Settings")
selected_classes = st.sidebar.multiselect(
    "Choose detection classes:",
    CLASS_NAMES,
    default=["person", "car", "truck"],
    help="Select one or more object classes for AI detection"
)
st.sidebar.info("**Tip:** Multi-class detection is supported for complex scenarios.")

st.sidebar.markdown("### About")
st.sidebar.markdown("""
**AutoVision Labeler** accelerates your computer vision workflow by:
- Automated object detection
- Multi-class recognition
- Instant analytics
- Export-ready annotations
""")

# ---------------------- MAIN CONTENT AREA ----------------------

# ---------------------- FILE UPLOAD SECTION ----------------------
st.markdown("### Upload Images")
uploaded_files = st.file_uploader(
    "Drag and drop your images here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload one or multiple images for batch processing"
)

if uploaded_files:
    results_data = []
    total_start = time.time()
    
    st.markdown("---")
    st.markdown("### Detection Results")
    
    # Determine number of columns based on number of images
    num_images = len(uploaded_files)
    if num_images == 1:
        num_cols = 1
    elif num_images == 2:
        num_cols = 2
    else:
        num_cols = 3
    
    # Create columns for processed images
    image_cols = st.columns(num_cols)
    col_index = 0

    for file in uploaded_files:
        bytes_data = file.read()
        np_img = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference
        start = time.time()
        results = model.predict(img_rgb, conf=0.3, verbose=False)
        end = time.time()
        infer_time = end - start

        detections = results[0]
        boxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()
        names = [model.names[int(c)] for c in classes]

        # Filter detections by selected objects
        filtered_indices = [i for i, name in enumerate(names) if name in selected_classes]
        filtered_boxes = boxes[filtered_indices] if len(filtered_indices) > 0 else []
        filtered_names = [names[i] for i in filtered_indices]

        if len(filtered_boxes) == 0:
            st.warning(f"No objects from selected classes detected in **{file.name}**")
            continue

        # Draw filtered boxes
        img_copy = img_rgb.copy()
        for box, label in zip(filtered_boxes, filtered_names):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), ACCENT_COLOR_BGR, 3)
            
            # Add label background
            label_text = label.upper()
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_copy, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), ACCENT_COLOR_BGR, -1)
            cv2.putText(img_copy, label_text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display image in its own column
        with image_cols[col_index % num_cols]:
            st.image(img_copy, caption=f"{file.name}", use_container_width=True)
            st.caption(f"**{infer_time:.3f}s** | **{len(filtered_names)} objects**")
        
        col_index += 1

        # Collect filtered results
        for i, box in enumerate(filtered_boxes):
            results_data.append({
                "image": file.name,
                "class": filtered_names[i],
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            })

    st.markdown("---")
    total_end = time.time()
    total_duration = total_end - total_start
    avg_time = total_duration / len(uploaded_files)
    
    
    # ---------------------- ANALYTICS DASHBOARD ----------------------
    st.markdown("<h3 style='text-align: center;'>Analytics Dashboard</h3>", unsafe_allow_html=True)
    
    df = pd.DataFrame(results_data)
    
    if len(df) > 0:
        # Metrics Row - Moved up before success message
        metric_cols = st.columns(4)
        
        avg_manual_time_per_box = 5
        total_boxes = len(df)
        estimated_manual_time = avg_manual_time_per_box * total_boxes
        time_saved_seconds = estimated_manual_time - total_duration
        time_saved_percent = 100 * (time_saved_seconds / estimated_manual_time) if estimated_manual_time > 0 else 0
        
        with metric_cols[0]:
            st.metric("Time Saved", f"{time_saved_percent:.1f}%", delta="vs Manual")
        
        with metric_cols[1]:
            st.metric("Total Detections", f"{total_boxes}", delta=f"{len(df['class'].unique())} classes")
        
        with metric_cols[2]:
            st.metric("Processing Speed", f"{avg_time:.2f}s", delta="per image")
        
        with metric_cols[3]:
            st.metric("Images Processed", f"{len(uploaded_files)}", delta="batch mode")
        
        st.success(f"**Processing Complete** | {len(uploaded_files)} images processed in {total_duration:.2f}s (avg: {avg_time:.3f}s/image)")
        
        st.markdown("---")
        
        # Data Visualization Section
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("#### Class Distribution Analysis")
            class_counts = df["class"].value_counts()
            
            # Create professional chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(class_counts.index, class_counts.values, color=ACCENT_COLOR, edgecolor=PRIMARY_COLOR, linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Professional styling
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#F8FAFC')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(BORDER_COLOR)
            ax.spines['bottom'].set_color(BORDER_COLOR)
            ax.tick_params(colors=TEXT_PRIMARY, labelsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
            
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Detection Count", fontsize=12, fontweight='600', color=TEXT_PRIMARY)
            plt.xlabel("Object Class", fontsize=12, fontweight='600', color=TEXT_PRIMARY)
            plt.title("Detected Objects by Class", fontsize=14, fontweight='700', color=TEXT_PRIMARY, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            st.markdown("#### Detection Data Sample")
            st.dataframe(
                df.head(15).style.set_properties(**{
                    'background-color': BG_LIGHT,
                    'color': TEXT_PRIMARY,
                    'border-color': BORDER_COLOR
                }),
                use_container_width=True,
                height=400
            )

        # ---------------------- DOWNLOAD SECTION ----------------------
        st.markdown("---")
        st.markdown("### Export Annotations")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            json_data = json.dumps(results_data, indent=2)
            st.download_button(
                "Download JSON",
                json_data,
                file_name="autovision_annotations.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                file_name="autovision_annotations.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            st.markdown(f"**Format:** YOLO Compatible  \n**Total Records:** {len(df)}")

    else:
        st.info("No detections matched the selected classes. Try adjusting your class selection.")

else:
    # Welcome State
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0;'>
        <h3 style='color: #64748B; font-weight: 400;'>Ready to Begin</h3>
        <p style='color: #94A3B8; font-size: 1.1rem;'>Upload images above to start AI-powered annotation</p>
    </div>
    """, unsafe_allow_html=True)