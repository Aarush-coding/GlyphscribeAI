import streamlit as st
import torch
from PIL import Image
import numpy as np
from model import MNISTV3
from preprocessing import preprocess_from_pil, preprocess_from_canvas, get_preview_image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="GlyphScribe AI",
    page_icon="🔱",
    layout="wide"
)

def theme(dark_mode):
    if dark_mode:
        sidebar_bg = '#141414'
        sidebar_text = "#e8e8e8"
        border = '#2a2a2a'
    else:
        sidebar_bg = "#ebebeb"
        sidebar_text = "#0f0f0f"
        border = "#d0d0d0"


    bg = "#f5f5f5"
    text = "#e3e3e3"
    subtext = "#666"
    card_bg = "#E7E7E7"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DynaPuff:wght@400..700&display=swap');

    html, body, [class*="css"] {{
        background-color: {bg} !important;
        color: {text} !important;
        font-family: 'DynaPuff' !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border};
    }}
    [data-testid="stSidebar"] * {{
        color: {sidebar_text} !important;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; }}
    h1, h2, h3 {{
        font-family: 'DynaPuff' !important;
        color: {text} !important;
    }}
    .stButton > button {{
        background: {text} !important;
        color: {bg} !important;
        border: none !important;
        border-radius: 4px !important;
        font-size: 0.8rem !important;
        padding: 0.6rem 1.5rem !important;
    }}
    .stButton > button:hover {{ opacity: 0.85 !important; }}
    hr {{ border-color: {border} !important; }}
    [data-testid="metric-container"] {{
        background: {card_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    [data-testid="stFileUploader"] {{
        background: {card_bg} !important;
        border: 1px dashed {border} !important;
        border-radius: 8px !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent;
        border-bottom: 1px solid {border};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {subtext} !important;
        font-size: 0.8rem !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {text} !important;
        border-bottom: 2px solid {text} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

with st.sidebar:
    st.markdown('# NAVIGATION')
    st.divider()
    pages = st.radio(
        label='',
        options=['HOME 🏠', 'MNIST DIGIT RECOGNITION', 'CREDITS & DOCS'],
        label_visibility='collapsed'
    )
    st.divider()
    st.session_state.dark_mode = st.toggle(
        'Dark mode',
        value=st.session_state.dark_mode
    )
    
@st.cache_resource
def load_model():
    model = MNISTV3(input_shape=1, hidden_units=10, output_shape=10)
    model.load_state_dict(torch.load('CNNmodel.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

theme(st.session_state.dark_mode)
if pages == 'HOME 🏠':
    st.markdown('# Welcome to GlyphScribe AI 👋')
    st.markdown('#### MNIST Digit Recognition')
    st.divider()
    st.markdown('''
    This is an app built using streamlit for my CS coursework,
    Here I trained a CNN to predict numbers you write! And most of all to understand Mahad's handwriting 😂
    ''')
    st.image('https://miro.medium.com/v2/resize:fit:1400/1*SfRJNb5dOOPZYEFY5jDRqA.png')
    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Model Accuracy", value="98.9%")
    with m2:
        st.metric(label="Training Images", value="60,000")
    with m3:
        st.metric(label="Test Images", value="10,000")
    st.divider()
    st.markdown(
        '''
        ### INSTRUCITONS
        1. **Draw or upload** a digit 
        2. if uploaded it must be **very clear and in a well lit room**
        3. A **prediciton** is returned
        '''
    )
elif pages == 'MNIST DIGIT RECOGNITION':
    st.markdown('# MNIST DIGIT RECOGNISER ')
    st.divider()
    
    tab1, tab2 = st.tabs(['Draw ✍🏼', 'upload 📁'])
    with tab1:
        st.markdown('## Draw your image')
        st.divider()
        st.markdown('Make sure to draw the digit clearly and fill the canvas as much as possible for better results!')
        canvas_result = st_canvas(
            fill_color = 'white',
            stroke_width = 25,
            stroke_color = 'white',
            background_color = 'black',
            height = 280,
            width = 280,
            drawing_mode = 'freedraw',
            key = 'canvas'
        )
        
        if st.button('Predict', key='canvas_input'):
            if canvas_result.image_data is not None:
                img = Image.fromarray(
                    canvas_result.image_data.astype('uint8'), 'RGBA'
                )
                tensor = preprocess_from_canvas(img)

                with torch.no_grad():
                    output = model(tensor)
                    prediction = output.argmax(dim=1).item()
                
                st.divider()
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown('#### Predicted Digit')
                    st.markdown(f"""
                    <div style='font-size: 6rem; font-weight: 700; 
                                font-family: monospace; text-align: center;'>
                        {prediction}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown("#### What the model sees")
                    preview = get_preview_image(img, source="canvas")
                    st.image(preview, width=200)
            else:
                st.warning('PLEASE DRAW SOMTHING FIRST')
    with tab2:
        st.markdown('## Upload an image')
        st.divider()
        uploaded = st.file_uploader(
            'Choose an image',
            type=['png', 'jpg', 'jpeg'],
            label_visibility='collapsed'
        )

        if uploaded is not None:
            img = Image.open(uploaded)
            tensor = preprocess_from_pil(img)
            with torch.no_grad():
                output = model(tensor)
                prediction = output.argmax(dim=1).item()

            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('#### Your image')
                st.image(img, width=200)
            with col2:
                st.markdown('#### What model sees')
                preview = get_preview_image(img, source='photo')
                st.image(preview, width=200)
            with col3:
                st.markdown('#### Prediction')
                st.markdown(f"""
                <div style='font-size: 6rem; font-weight: 700;
                            font-family: monospace; text-align: center;'>
                    {prediction}
                </div>
                """, unsafe_allow_html=True)
elif pages == 'CREDITS & DOCS':
    st.markdown('# CREDITS & DOCS')
    st.divider()
    with open("/Users/Aarush/Desktop/Python Repositrory/Handwritng_analyisis_coursework/MAIN/GlyphScribe AI.docx", "rb") as file:
        btn = st.download_button(
            label="Download Word Document",
            data=file,
            file_name="GLYPHSCRIBE_AI_DOWNLOAD.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )


