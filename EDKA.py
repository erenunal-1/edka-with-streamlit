# Libraries
import streamlit as st
from streamlit_option_menu import option_menu
import os
from streamlit_lottie import st_lottie
import pandas as pd
from ydata_profiling import ProfileReport
import sweetviz as sv
import json
import pygwalker as pyg
import streamlit.components.v1 as stc

# Wide mode default value has been changed.
st.set_page_config(layout="wide")

# Dataset
if os.path.exists("./dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)

# Lottie
@st.cache_data
def load_lottiefile(path: str):
    with open(path) as f:
        data = json.load(f)
        return data

lottie_file = "Animation.json"
lottie_json = load_lottiefile(lottie_file) 

# Sidebar
with st.sidebar:
    st_lottie(lottie_json)

    selected = option_menu("EDKA", ["Upload", "PyGWalker", "Pandas Profiling", "SweetViz",
                            "Rapid Model Builder", "About"],  icons=["cloud-upload", "", "", "",
                            "download", "info-circle"], menu_icon="list-task", default_index=0)
    st.markdown(":blue[**For any information or just a quick Hi.**] "
                "[![LinkedIn](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png"
                "/25px-LinkedIn_logo_initials.png)](https://www.linkedin.com/in/eren-unal/)")

# Upload csv
if selected == "Upload":
    st.header("Upload Your Dataset", divider="rainbow")
    file = st.file_uploader("")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)

# PyGWalker
if selected == "PyGWalker":
    st.header("Autopilot for Exploratory Data Analysis with PyGWalker", divider="rainbow")
    df = pd.read_csv("dataset.csv")
    pyg_walker_html = pyg.walk(df, return_html=True)
    stc.html(pyg_walker_html, scrolling=True, height=1000)

# ydataprofiling
if selected == "Pandas Profiling":
    st.header("Automated Data Visualization with Pandas Profiling", divider="rainbow")
    df = pd.read_csv("dataset.csv")
    profile = ProfileReport(df)
    profile_path = "profile_report.html"
    profile.to_file(output_file=profile_path)
    with open(profile_path, "r") as f:
        html = f.read()
    st.components.v1.html(html, width=1600, height=800, scrolling=True)

# sweetviz
if selected == "SweetViz":
    st.header("Automated Data Visualization with SweetViz", divider="rainbow")
    df = pd.read_csv("dataset.csv")
    report = sv.analyze(df)
    st.components.v1.html(report.show_html(), width=1600, height=800, scrolling=True)

# Rapid Model Builder
if selected == "Rapid Model Builder":
    st.header("Automated Rapid Model Builder", divider="rainbow")
    chosen_target = st.selectbox("Choose the Target Column", df.columns, help="The target column is the variable you "
                                                                              "want to predict.")
    chosen_problem = st.selectbox("Choose the Problem Type", ["Classification", "Regression"])
    if chosen_problem == "Classification":
        from pycaret.classification import setup, compare_models, pull, save_model, load_model
    elif chosen_problem == "Regression":
        from pycaret.regression import setup, compare_models, pull, save_model, load_model

    if st.button("Run Modelling"):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        rapidmodel = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(rapidmodel, "rapidmodel")
        with open("rapidmodel.pkl", "rb") as f:
            st.download_button("Download Model", f.read(), file_name="rapidmodel.pkl", mime="application/octet-stream")

# About
if selected == "About":

    st.header("About", divider="rainbow")
    st.info("We aim to convey your data not only with figures but also through compelling stories that reflect the real"
            " world behind those figures. Explore the enchanting world of data science with us. From data visualization"
            " to machine learning models, tailor-made solutions await you right here. Discover the magical libraries"
            " within the **EDKA** world and unleash the magic!")

    st.header("📈 PyGWalker", divider="rainbow")
    st.info(":blue[**PyGWalker :**] PyGWalker transforms your data into interactive visualization apps with a line of"
            " code, enabling effortless sharing with one click. It also offers a range of features designed to "
            "simplify data analysis, always ensuring scalability. With Data Painter, you can refine your data swiftly"
            " using an eraser to remove outliers, clusters and complex patterns within seconds. You can create new"
            " variables, labels, or features seamlessly without disrupting your analysis workflow. Data Painter allows"
            " you to create and incorporate new features into your analytical views in real-time. Dive into insights"
            " with ease and enjoy a smoother analytical journey with PyGWalker.")

    video_file = open("PyGWalker Trailer.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.header("📊 Pandas Profiling", divider="rainbow")
    st.info(":blue[**Pandas Profiling :**] Standing as a beacon of insight in the field of data analysis, "
            "Pandas Profiling offers a feature-rich treasure trove to illuminate the depths of your data set. "
            "With just a few lines of code, it unveils valuable information crucial for efficient data exploration "
            "and analysis by generating a comprehensive report. The beauty of Pandas Profiling lies in its simplicity"
            " and efficiency. With minimal effort, it saves valuable time and effort in the analytical journey by "
            "crafting a detailed narrative of your dataset. Take a plunge into the realm of Pandas Profiling and "
            "embark on a discovery journey.")

    video_file = open("Pandas Profiling Trailer.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.header("📉 SweetViz", divider="rainbow")
    st.info(":blue[**SweetViz :**] Unlock the secrets of your data swiftly and comprehensively. With just a few clicks,"
            " reveal distributions, missing values, correlations, and more. Initiate your exploration with SweetViz "
            "and unveil the magic within your datasets!")

    video_file = open("SweetViz Trailer.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.header("🤖 Rapid Model Builder", divider="rainbow")
    st.info(":blue[**Rapid Model Builder :**] Initially obscure and complex data shines like stars with the guidance "
            "of Rapid Model Builder, leading you in the right direction. Depending on the type of problem you choose, "
            "it automates the necessary steps to predict the target variable in your dataset and evaluates various "
            "models. Afterwards, it selects the most dazzling model among the options. ")

    video_file = open("Rapid Model Builder Trailer.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.header(" Custom Model Builder", divider="rainbow")
    st.error(":blue[**Custom Model Builder :**] Custom Model Builder opens the doors to the magical realm of data "
            "science, granting you the opportunity to unearth the potential hidden within your datasets. "
            "Guiding you at every step, we offer customized machine learning models tailored to your unique needs. "
            "Join us on this thrilling adventure and let's together elevate Custom Model Builder. Feel free to get "
            "in touch.")
