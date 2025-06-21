import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.inference import generate_output

st.set_page_config(page_title="Code Assistant", layout="wide")
st.title("💻 Code Generation & Explanation Assistant")

option = st.selectbox("Choose task", ["Code Generation", "Code Explanation"])

prompt = st.text_area("Enter prompt or code")

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = generate_output(prompt)
        st.subheader("Output")
        st.code(output, language="python")
