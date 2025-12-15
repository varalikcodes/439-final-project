import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import json

st.set_page_config(layout="wide", page_title="Reviews Dashboard")

DATA_PATH = "Arts_Crafts_and_Sewing.jsonl"

@st.cache_data
def load_json_data_fast(path, sample_size=None):
    """Load JSONL data efficiently with minimal memory overhead."""
    data = []
    count = 0
    
    # Count total lines quickly
    with open(path, 'r') as f:
        total = sum(1 for _ in f)
    
    step = max(1, total // sample_size) if sample_size else 1
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i % step == 0:
                try:
                    data.append(json.loads(line))
                    count += 1
                    if sample_size and count >= sample_size:
                        break
                except:
                    pass
    
    df = pd.DataFrame(data)
    # Keep only essential columns
    cols_to_keep = [col for col in ['rating', 'text', 'title', 'helpful_vote'] if col in df.columns]
    df = df[cols_to_keep]
    
    # Drop zero ratings
    df = df[df['rating'] != 0]
    
    # Create combined review column
    df['rev'] = df['text'].astype(str) + ' ' + df['title'].astype(str)
    
    return df

st.title("Amazon Arts, Crafts & Sewing — Reviews Dashboard")

# Initialize session state for data loading
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load options in columns
col_btn, col_sample = st.columns([2, 1])

with col_btn:
    load_button = st.button("Load Data", key='load_data_button', use_container_width=True)

with col_sample:
    sample_mode = st.selectbox(
        "Load mode:",
        ["Sample (fastest)", "Full Dataset"],
        help="Sample mode loads ~2000 reviews for quick exploration"
    )

sample_size = 2000 if sample_mode == "Sample (fastest)" else None

# Load data button action
if load_button:
    with st.spinner("Loading data..."):
        df = load_json_data_fast(DATA_PATH, sample_size=sample_size)
        st.session_state.df = df
        st.session_state.data_loaded = True
    
    msg = f"Loaded {len(df):,} reviews"
    if sample_size:
        msg += " (sample mode)"
    st.success(msg)

# If data is not loaded, stop here
if not st.session_state.data_loaded:
    st.info("Click 'Load Data' to begin exploring reviews")
    st.stop()

# Data is loaded, show everything below
df = st.session_state.df

# Main title section
st.markdown("### Filter Controls")
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 1, 1, 1])

with filter_col1:
    stop_add = st.text_input("Extra stopwords (comma separated):", "use,used,using,br,color,one,bought,work,colors,like,product,make,time")

with filter_col2:
    show_proportional = st.checkbox("Downsample 5-star", value=False)

with filter_col3:
    if show_proportional:
        five_pct = st.slider("Keep %", 1, 100, 20, key='five_pct') / 100.0
    else:
        five_pct = 0.2

with filter_col4:
    pass  # spacer

st.markdown("---")

# Filters for plots
st.markdown("### Plot Filters")
filter_row1_col1, filter_row1_col2, filter_row1_col3 = st.columns([2, 1, 1])

with filter_row1_col1:
    star_filter = st.multiselect("Filter by star rating:", sorted(df['rating'].unique()), default=sorted(df['rating'].unique()), key='star_filter')

with filter_row1_col2:
    max_len_val = int(df['text'].astype(str).str.len().max())
    min_len = st.number_input("Min review length:", 0, max_len_val, 0, key='min_len')

with filter_row1_col3:
    max_len = st.number_input("Max review length:", 0, max_len_val, max_len_val, key='max_len')

# Apply filters
df_filtered = df[(df['rating'].isin(star_filter)) & 
                 (df['text'].astype(str).str.len() >= min_len) & 
                 (df['text'].astype(str).str.len() <= max_len)]

st.success(f"Showing {len(df_filtered)} of {len(df)} reviews")

st.markdown("---")

# Main layout: top row histograms, middle row scatter plots, bottom row wordclouds
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Star counts")
    rating = df_filtered['rating']
    fig1, ax1 = plt.subplots()
    x = sorted(rating.unique())
    y = rating.value_counts(sort=False).reindex(x).fillna(0)
    sns.barplot(x=list(x), y=y.values, palette='Blues', ax=ax1)
    ax1.set_xlabel('Star rating')
    ax1.set_ylabel('Number of reviews')
    st.pyplot(fig1)

with col2:
    st.subheader("Proportional star counts")
    if show_proportional:
        df_prop = df_filtered.copy()
        drop_idx = df_prop[df_prop['rating'] == 5].index
        num = int(len(drop_idx) * (1.0 - five_pct))
        if num > 0:
            ran_idx = np.random.choice(drop_idx, size=num, replace=False)
            df_prop = df_prop.drop(ran_idx)
    else:
        df_prop = df_filtered

    fig2, ax2 = plt.subplots()
    rating2 = df_prop['rating']
    x2 = sorted(rating2.unique())
    y2 = rating2.value_counts(sort=False).reindex(x2).fillna(0)
    sns.barplot(x=list(x2), y=y2.values, palette='Greens', ax=ax2)
    ax2.set_xlabel('Star rating')
    ax2.set_ylabel('Number of reviews')
    st.pyplot(fig2)

# Scatter plots
col3, col4 = st.columns([1,1])
with col3:
    st.subheader("Review length vs Stars")
    fig3, ax3 = plt.subplots()
    xlen = df_filtered['text'].astype(str).str.len()
    ax3.scatter(xlen, df_filtered['rating'], alpha=0.3)
    ax3.set_xlabel('review length')
    ax3.set_ylabel('stars')
    st.pyplot(fig3)

with col4:
    st.subheader("Helpful votes vs Stars")
    fig4, ax4 = plt.subplots()
    xhelp = df_filtered.get('helpful_vote', pd.Series(0, index=df_filtered.index))
    ax4.scatter(xhelp, df_filtered['rating'], alpha=0.3)
    ax4.set_xlabel('helpful votes')
    ax4.set_ylabel('stars')
    st.pyplot(fig4)

# Wordclouds
st.markdown("---")
st.subheader("Wordclouds by Star Rating")

st.write("Stopwords used for wordclouds:")
st.write(stop_add)

st_columns = st.columns(5)
extra_stops = set([w.strip().lower() for w in stop_add.split(',') if w.strip()])
stops = set(text.ENGLISH_STOP_WORDS) | extra_stops

for i, rating_val in enumerate([1,2,3,4,5]):
    col = st_columns[i]
    with col:
        subset = df_filtered[df_filtered['rating'] == rating_val]
        text_blob = ' '.join(subset['text'].astype(str).tolist())
        if len(text_blob.strip()) == 0:
            st.write(f"No text for {rating_val}-star")
            continue
        wc = WordCloud(width=400, height=300, background_color='white', stopwords=stops).generate(text_blob)
        fig_wc, ax_wc = plt.subplots(figsize=(4,3))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title(f"{rating_val} Star")
        st.pyplot(fig_wc)

st.markdown("---")

# Training and prediction section
with st.expander("Train Model & Predict Review Scores", expanded=True):
    st.subheader('Train Models')
    
    col_train_a, col_train_b = st.columns([1, 1])
    with col_train_a:
        test_size = st.slider('Test fraction', 0.05, 0.5, 0.2, key='test_fraction')
    with col_train_b:
        st.write("**Model:** MultinomialNB")
    
    # Initialize trained models tracking in session state
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    model_choice = 'MultinomialNB'
    # Check if this model is already trained
    is_trained = model_choice in st.session_state.trained_models
    
    train_disabled = is_trained
    if st.button("Train Model", key='train_button', disabled=train_disabled):
        with st.spinner("Training model..."):
            X = df_filtered['rev']
            y = df_filtered['rating']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

            stops_list = list(text.ENGLISH_STOP_WORDS) + list(extra_stops)
            vectorizer = CountVectorizer(stop_words=stops_list)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            model = MultinomialNB().fit(X_train_vec, y_train)
            
            y_pred = model.predict(X_test_vec)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Store trained model in session state
            st.session_state.trained_models[model_choice] = {
                'model': model,
                'vectorizer': vectorizer,
                'report': report,
                'cm': cm
            }
            
        st.success(f"{model_choice} model trained successfully!")
        st.rerun()
    
    if is_trained:
        st.info(f"{model_choice} is trained. Use it for predictions below.")
        
        # Display confusion matrix in compact form
        with st.expander(f"View {model_choice} Details", expanded=False):
            model_data = st.session_state.trained_models[model_choice]
            
            col_cm, col_report = st.columns([1, 1])
            with col_cm:
                st.markdown("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                sns.heatmap(model_data['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                st.pyplot(fig_cm, use_container_width=False)
            
            with col_report:
                st.markdown("**Classification Report**")
                st.text(model_data['report'])
    
    st.markdown("---")
    st.subheader("Predict Review Score")
    
    # Select which trained model to use for prediction
    if st.session_state.trained_models:
        trained_model_names = list(st.session_state.trained_models.keys())
        selected_model = st.selectbox("Select trained model for prediction:", trained_model_names, key='predict_model_select')
        
        user_review = st.text_area("Enter a review to predict its rating:", placeholder="Type your review here...", key='review_input')
        
        if st.button("Predict Rating", key='predict_button'):
            if user_review.strip():
                model_data = st.session_state.trained_models[selected_model]
                vectorizer = model_data['vectorizer']
                model = model_data['model']
                
                review_vec = vectorizer.transform([user_review])
                predicted_rating = model.predict(review_vec)[0]
                
                st.write("---")
                st.metric("Predicted Rating", f"{predicted_rating} stars")
                st.write(f"**Model used:** {selected_model}")
            else:
                st.warning("Please enter a review text.")
    else:
        st.info("Train at least one model first to make predictions!")