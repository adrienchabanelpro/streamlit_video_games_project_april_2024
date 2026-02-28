# style.py
import streamlit as st


def apply_style() -> None:
    """Inject global CSS for the retro neon dark theme."""
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Tiny5&display=swap" rel="stylesheet">
        <style>
        /* ---- Retro Neon Dark Theme ---- */

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1A1A2E;
            border-right: 2px solid #00FFCC;
        }
        [data-testid="stSidebar"] * {
            color: #E0E0E0;
        }

        /* Headings — neon cyan with glow */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Tiny5', sans-serif !important;
            color: #00FFCC !important;
            text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
        }

        /* Body text */
        .stMarkdown, .stText, p, li, span {
            color: #E0E0E0;
        }

        /* Links */
        a {
            color: #FF6EC7 !important;
        }
        a:hover {
            color: #FFFF00 !important;
            text-shadow: 0 0 8px rgba(255, 255, 0, 0.6);
        }

        /* Buttons — neon pink accent */
        .stButton > button {
            background-color: transparent;
            color: #FF6EC7;
            border: 2px solid #FF6EC7;
            font-family: 'Press Start 2P', cursive;
            font-size: 12px;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            background-color: #FF6EC7;
            color: #0D0D0D;
            box-shadow: 0 0 15px rgba(255, 110, 199, 0.6);
        }

        /* Metrics — neon glow */
        [data-testid="stMetric"] {
            background-color: #1A1A2E;
            border: 1px solid #00FFCC;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 0 8px rgba(0, 255, 204, 0.2);
        }
        [data-testid="stMetricValue"] {
            color: #00FFCC !important;
            text-shadow: 0 0 6px rgba(0, 255, 204, 0.4);
        }
        [data-testid="stMetricLabel"] {
            color: #E0E0E0 !important;
        }

        /* Selectbox / number input */
        .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #E0E0E0 !important;
        }

        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #00FFCC;
            border-radius: 4px;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            color: #E0E0E0;
        }
        .stTabs [aria-selected="true"] {
            color: #00FFCC !important;
            border-bottom-color: #00FFCC !important;
        }

        /* Divider lines */
        hr {
            border-color: #333366;
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: #FF6EC7 !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 1px dashed #00FFCC;
            border-radius: 8px;
            padding: 10px;
        }

        /* Warning / info / error boxes */
        .stAlert {
            border-radius: 8px;
        }

        /* Navigation items */
        [data-testid="stSidebarNav"] a {
            color: #E0E0E0 !important;
        }
        [data-testid="stSidebarNav"] a:hover {
            color: #00FFCC !important;
        }

        /* ---- 4a: Enhanced CSS — Expanders ---- */
        .streamlit-expanderHeader {
            background-color: #1A1A2E !important;
            border: 1px solid #00FFCC !important;
            border-radius: 8px;
            color: #E0E0E0 !important;
            box-shadow: 0 0 8px rgba(0, 255, 204, 0.25);
        }
        .streamlit-expanderContent {
            background-color: #12122A !important;
            border: 1px solid #00FFCC !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px;
        }
        details[data-testid="stExpander"] {
            background-color: #1A1A2E;
            border: 1px solid #00FFCC;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 255, 204, 0.2);
        }
        details[data-testid="stExpander"] summary {
            color: #E0E0E0 !important;
        }
        details[data-testid="stExpander"] > div {
            background-color: #12122A;
        }

        /* ---- Text inputs and text areas ---- */
        .stTextInput input,
        .stTextArea textarea {
            background-color: #1A1A2E !important;
            color: #E0E0E0 !important;
            border: 1px solid #333366 !important;
            border-radius: 6px;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: #00FFCC !important;
            box-shadow: 0 0 8px rgba(0, 255, 204, 0.4) !important;
            outline: none !important;
        }

        /* ---- Sliders: dark track, cyan glow thumb ---- */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #00FFCC !important;
            box-shadow: 0 0 10px rgba(0, 255, 204, 0.6);
            border: 2px solid #00FFCC !important;
        }
        .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
            background-color: #1A1A2E !important;
        }
        .stSlider [data-baseweb="slider"] > div > div:first-child {
            background-color: #333366 !important;
        }
        .stSlider [data-baseweb="slider"] > div > div:nth-child(2) {
            background: linear-gradient(90deg, #00FFCC, #FF6EC7) !important;
        }

        /* ---- Radio buttons and checkboxes: neon accent ---- */
        .stRadio label,
        .stCheckbox label {
            color: #E0E0E0 !important;
            transition: color 0.2s;
        }
        .stRadio label:hover,
        .stCheckbox label:hover {
            color: #00FFCC !important;
        }
        .stRadio [role="radiogroup"] label[data-checked="true"],
        .stRadio [role="radiogroup"] label[aria-checked="true"] {
            color: #00FFCC !important;
            text-shadow: 0 0 6px rgba(0, 255, 204, 0.4);
        }
        .stCheckbox input:checked + label {
            color: #00FFCC !important;
            text-shadow: 0 0 6px rgba(0, 255, 204, 0.4);
        }
        /* Custom radio/checkbox indicator color */
        .stRadio [data-baseweb="radio"] div:first-child {
            border-color: #00FFCC !important;
        }
        .stRadio [data-baseweb="radio"] div:first-child::after {
            background-color: #00FFCC !important;
        }
        .stCheckbox [data-baseweb="checkbox"] span {
            border-color: #00FFCC !important;
        }
        .stCheckbox [data-baseweb="checkbox"] span[aria-checked="true"] {
            background-color: #00FFCC !important;
            border-color: #00FFCC !important;
        }

        /* ---- Progress bars: gradient cyan to pink ---- */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #00FFCC, #FF6EC7) !important;
            border-radius: 4px;
        }
        .stProgress > div > div {
            background-color: #1A1A2E !important;
            border-radius: 4px;
        }

        /* ---- Custom scrollbar ---- */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #0D0D0D;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb {
            background: #00FFCC;
            border-radius: 5px;
            box-shadow: 0 0 6px rgba(0, 255, 204, 0.3);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #33FFD6;
        }

        /* ---- Neon pulse keyframes for headings ---- */
        @keyframes neonPulse {
            0%, 100% {
                text-shadow: 0 0 10px rgba(0, 255, 204, 0.5),
                             0 0 20px rgba(0, 255, 204, 0.2);
            }
            50% {
                text-shadow: 0 0 16px rgba(0, 255, 204, 0.8),
                             0 0 30px rgba(0, 255, 204, 0.4);
            }
        }
        h1, h2 {
            animation: neonPulse 3s ease-in-out infinite;
        }

        /* ---- Retro scanline overlay on main content area ---- */
        [data-testid="stAppViewContainer"]::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 0, 0, 0.03) 0px,
                rgba(0, 0, 0, 0.03) 1px,
                transparent 1px,
                transparent 3px
            );
        }

        /* ---- 4d: Mobile-responsive layout ---- */
        @media screen and (max-width: 768px) {
            /* Reduce heading font sizes */
            h1 {
                font-size: 1.4rem !important;
            }
            h2 {
                font-size: 1.2rem !important;
            }
            h3 {
                font-size: 1rem !important;
            }
            h4, h5, h6 {
                font-size: 0.9rem !important;
            }

            /* Make sidebar narrower */
            [data-testid="stSidebar"] {
                min-width: 180px !important;
                max-width: 220px !important;
            }
            [data-testid="stSidebar"] img {
                max-width: 120px !important;
            }

            /* Adjust chart container widths */
            .stPlotlyChart,
            [data-testid="stPlotlyChart"],
            .stDataFrame {
                width: 100% !important;
                max-width: 100% !important;
                overflow-x: auto;
            }

            /* Stack metric cards vertically */
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 8px !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            /* Reduce padding and margins */
            .block-container {
                padding: 0.5rem 0.8rem !important;
            }
            [data-testid="stMetric"] {
                padding: 8px !important;
            }
            .stButton > button {
                font-size: 10px !important;
                padding: 6px 12px !important;
            }

            /* Arcade screen overlay adjustments */
            .arcade-screen {
                font-size: 14px !important;
                top: -200px !important;
                width: 80% !important;
                height: 140px !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
