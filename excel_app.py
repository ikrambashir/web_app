import streamlit as st
import pandas as pd

st.title("Excel App")

df = pd.read_csv("data.csv")
st.header("Existing file")
st.write(df)
# st.table(df)

st.sidebar.header("Options")
options_form = st.sidebar.form("Options_form")
user_name = options_form.text_input("Name")
user_age = options_form.number_input("Age")
add_data = options_form.form_submit_button()
if add_data:
    new_data = {"name": user_name, "age": int(user_age)}
    df = df.append(new_data, ignore_index=True)
    df.to_csv("data.csv", index=False)