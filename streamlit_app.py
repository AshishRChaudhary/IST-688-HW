import streamlit as st

st.set_page_config(page_title="HW Manager")
st.sidebar.title("HW Manager")

hw1_page = st.Page("HW/HW1.py", title="HW 1")
hw2_page = st.Page("HW/HW2.py", title="HW 2", default=True)
hw3_page = st.Page("HW/HW3.py", title="HW 3")
hw4_page = st.Page("HW/HW4.py", title="HW 4")
hw5_page = st.Page("HW/HW5.py", title="HW 5")
hw7_page = st.Page("HW/HW7.py", title="HW 7")

pg = st.navigation([hw1_page, hw2_page, hw3_page, hw4_page, hw5_page, hw7_page])
pg.run()
