import streamlit as st

st.set_page_config(page_title="HW Manager")
st.sidebar.title("HW Manager")

hw1_page = st.Page("HW/HW1.py", title="HW 1")
hw2_page = st.Page("HW/HW2.py", title="HW 2", default=True)

pg = st.navigation([hw1_page, hw2_page])
pg.run()
