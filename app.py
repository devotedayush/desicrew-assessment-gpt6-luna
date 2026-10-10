"""Run with: streamlit run app.py --server.address 127.0.0.1"""
import streamlit as st
from desicrew.ui import inventory_page, support_page, documents_page, sidebar_settings

st.set_page_config(page_title="DesiCrew · AI workbench", page_icon="◈", layout="wide")
sidebar_settings()
page = st.navigation([
    st.Page(inventory_page, title="Inventory code agent", icon="📊", url_path="inventory", default=True),
    st.Page(support_page, title="Document support", icon="💬", url_path="support"),
    st.Page(documents_page, title="Document review", icon="🔎", url_path="documents"),
])
page.run()
