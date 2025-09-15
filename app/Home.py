import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd

color_scheme = {
    "background": "#1F2937",       # Dark Gray
    "surface": "#374151",          # Charcoal
    "primary": "#6366F1",          # Indigo
    "secondary": "#2DD4BF",        # Teal
    "accent": "#FDB515",           # Amber
    "text_primary": "#F9FAFB",     # Light Gray
    "text_muted": "#9CA3AF",       # Soft Gray
}

#Page Configuration
st.set_page_config(
    page_title="TaskMiner: Tasks made easy!",
    page_icon="🤖",
    layout="wide"
)

#Initialize Authenticator
if 'authenticator' not in st.session_state:
    with open('users/config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    st.session_state.authenticator = authenticator
else:
    authenticator = st.session_state.authenticator

#Login Widget
try:
    authenticator.login('sidebar')
except Exception as e:
    st.error(e)

if st.session_state.get('authentication_status'):
    st.sidebar.markdown(f'### Welcome *{st.session_state.get("name")}*,')
    authenticator.logout(location='sidebar')

#Page Content
st.markdown("""<h1 style="text-align: center;">TaskMiner</h1>""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        background-color: #374151;
        margin-bottom: 20px;
        padding: 10px 50px;
        border-radius: 20px;
        width: 100%;
        display: inline-block;
        color: #F9FAFB;">
        <h2> What exactly is it?</h2>
        TaskMiner is an AI task extraction tool that facilitates task tracking through it's ability to
        detect tasks from incoming emails.
        TaskMiner leverages LLM capabilities to accurately identify tasks by bringing in relevant emails into
        context from threads and previous conversations.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background-color: #374151;
        margin-bottom: 20px;
        padding: 10px 50px;
        border-radius: 25px;
        width: 100%;
        display: inline-block;
        color: #F9FAFB;">
        <h2> Why is it unique?</h2>
        TaskMiner sits in the gap between manual input productivity tools and formal ticketing systems, 
        offering lightweight automation for unstructured environments. With the rise of LLMs and practical NLP techniques, 
        we now have the tools to extract meaningful task data from unstructured emails reliably.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background-color: #374151;
        margin-bottom: 20px;
        padding: 10px 50px;
        border-radius: 25px;
        width: 100%;
        display: inline-block;
        color: #F9FAFB;">
        <h2> How do I get started?</h2>
        <ol>
            <li>Login using your <i>Username</i> and <i>Password</i></li>
            <li>Navigate to the <b><i>Dashboard</i></b> screen using the sidebar</li>
            <li>Upload your emails as EML files</li>
            <li>✅Accept or ❌Decline tasks through your <b><i>Task Decision Portal</i></b></li>
            <li>Track tasks through the <b><i>Task Manager</i></b> and mark tasks as completed✅</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)
