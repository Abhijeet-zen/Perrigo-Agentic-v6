"""
main.py

Entry point for the Multi-Agent AI system.
Prompts the user for API key before launching UI and agents.
"""

import os
import logging
import streamlit as st
from config import setup_logging
from src.ui import run_ui
from PIL import Image
from src.utils.openai_api import get_supervisor_llm

setup_logging()

def main():
    """
    Initializes the application and starts the UI.
    Ensures API key is provided before starting any agent processes.
    """
    # Initialize session state for sidebar visibility
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False

    # Set initial sidebar state
    st.set_page_config(page_title="GenAI Answer Bot",page_icon="📊",layout="wide",initial_sidebar_state="collapsed" if st.session_state.sidebar_collapsed else "expanded")

    st.markdown("""
        <style>
            section.main > div:first-child {
                padding-top: 3rem;
                max-width: 100rem !important;
            }
            [data-testid="stSidebar"] {
                width: 300px !important;
            }
        </style>
        """, unsafe_allow_html=True)

    try:
        logo = Image.open("Images/perrigo-logo.png")
        st.sidebar.image(logo, width=80)
    except Exception:
        st.sidebar.error("Logo image not found.")

    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password",
                                   key="api_key_input")

    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar to continue.")
        st.markdown("""
               <style>
                   [data-testid="collapsedControl"] {
                       display: block !important;
                   }
               </style>
               """, unsafe_allow_html=True)
        return

    if api_key and not st.session_state.sidebar_collapsed:
        st.session_state.sidebar_collapsed = True
        st.rerun()

    os.environ["OPENAI_API_KEY"] = api_key
    st.session_state["OPENAI_API_KEY"] = api_key

    try:
        st.session_state["llm"] = get_supervisor_llm(api_key)
    except ValueError as e:
        st.error(f"❌ Invalid API Key: {e}")
        return

    logging.info("✅ OpenAI API Key set. Starting the Multi-Agent AI System...")

    try:
        run_ui()
    except Exception as e:
        logging.error(f"Error starting the UI: {e}")
        st.error(f"An error occurred while launching the application: {e}")


if __name__ == "__main__":
    main()
