import pickle
import streamlit as st
from PIL import Image
import re
import nltk
nltk.download('punkt')
from nltk import tokenize
from sentence_transformers import SentenceTransformer


def collapse_dots(input):
    # Collapse sequential dots
    input = re.sub("\.+", ".", input)
    # Collapse dots separated by whitespaces
    all_collapsed = False
    while not all_collapsed:
        output = re.sub(r"\.(( )*)\.", ".", input)
        all_collapsed = input == output
        input = output
    return output

def process_text(input):
    if isinstance(input, str):
        input = " ".join(tokenize.sent_tokenize(input))
        input = re.sub(r"http\S+", "", input)
        input = re.sub(r"\n+", ". ", input)
        for symb in ["!", ",", ":", ";", "?"]:
            input = re.sub(rf"\{symb}\.", symb, input)
        input = re.sub("[^а-яА-Яa-zA-Z0-9!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ё]+", " ", input)
        input = re.sub(r"#\S+", "", input)
        input = collapse_dots(input)
        input = input.strip()
        # input = input.lower()
    return input

def load_model():
    model = pickle.load(open("model/model.pkl", "rb"))
    return model

def load_vectorizer():
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    return vectorizer

def final_pre_process_text(text, vectorizer):
    model_input = [process_text(text)]
    sent_tr = SentenceTransformer('all-MiniLM-L6-v2',device="cuda")
    model_input = sent_tr.encode(model_input)
    model_input = vectorizer.transform(model_input)
    return model_input


def main():
    st.header("Final Project: Troll Tweet Detection")
    clf = load_model()
    vectorizer = load_vectorizer()
    selected_page = st.sidebar.radio("Choose page:", ["Model", "EDA"])
    if selected_page == "Model":
        in_txt = st.text_input("Please, put your text here...")
        if in_txt:
            st.write(f"Text before pre-processing: {in_txt}")
            clean_txt = final_pre_process_text(in_txt, vectorizer)
            st.write(f"Text before pre-processing: {clean_txt}")
            st.write(f"Model output: {clf.predict_proba(clean_txt)[:, 1][0]}")

    elif selected_page == "EDA":
        image_name = st.selectbox("Select type of EDA", ["Word Cloud of '0' class", 
                                                    "Word Cloud of '1' class",
                                                    "Word Cloud of hashtages for '0' class",
                                                    "Word Cloud of hashtages for '1' class",
                                                    "Word Cloud of mentions for '0' class",
                                                    "Word Cloud of mentions for '1' class"                                                    
                                                    ]
                                                )
        image_path_dict = {
            "Word Cloud of '0' class": "images/wc_0_class.png", 
            "Word Cloud of '1' class": "images/wc_1_class.png",
            "Word Cloud of hashtages for '0' class": "images/wc_hshtg_0_class.png",
            "Word Cloud of hashtages for '1' class": "images/wc_hshtg_1_class.png",
            "Word Cloud of mentions for '0' class": "images/wc_mentions_0_class.png",
            "Word Cloud of mentions for '1' class": "images/wc_mentions_1_class.png"  
        }

        if image_name:
            st.subheader(image_name)
            image = Image.open(image_path_dict.get(image_name))
            st.image(image, caption=image_name)



if __name__ == "__main__":
    main()
