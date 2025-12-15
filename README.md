# Amazon Arts, Crafts & Sewing — Reviews Dashboard

An interactive Streamlit dashboard for exploring and analyzing product reviews. Visualize trends, generate word clouds, and train a machine learning model to predict review scores.

## Features

### Interactive Visualizations
- **Star Rating Distribution** — Bar chart showing review counts at each rating level
- **Proportional Star Count** — Optionally downsample 5-star reviews for balanced analysis
- **Review Length vs Stars** — Scatter plot showing the relationship between review length and rating
- **Helpful Votes vs Stars** — Scatter plot of how helpful reviews correlate with ratings
- **Word Clouds** — Visual word frequency for 1-star through 5-star reviews with custom stopwords

### Smart Filters
- **Filter by Star Rating** — Choose which ratings to display (multiselect)
- **Filter by Review Length** — Set minimum and maximum review length
- **Custom Stopwords** — Remove common words from word cloud generation
- **Live Count** — Always see how many reviews match your filters

### Machine Learning
- **Fast Training** — MultinomialNB model trains quickly on any dataset size
- **One-Time Training** — Train once, then reuse for multiple predictions
- **Confusion Matrix** — View model performance metrics
- **Review Score Prediction** — Type a review and get an instant star rating prediction!

## Setup

### Prerequisites
- Python 3.7+
- `Arts_Crafts_and_Sewing.jsonl` in the same folder as the app

### Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

The dashboard will open at `http://localhost:8501` in your browser.

## How to Use

1. **Load Data** — Click the "Load Data" button to start
   - Choose "Sample (fastest)" for quick exploration (~2,000 reviews)
   - Choose "Full Dataset" to load all reviews
2. **Explore Visualizations** — Use filters to focus on specific reviews and see real-time chart updates
3. **Adjust Filters** — Filter by rating, review length, and custom stopwords
4. **Train Model** — Expand "Train Model & Predict Review Scores" and click "Train Model"
5. **Make Predictions** — Once trained, enter any review text to predict its star rating!

## Files

- `streamlit_app.py` — Main dashboard application
- `requirements.txt` — Python package dependencies
- `README.md` — This file

## Performance Notes

- **Sample Mode** — Loads ~2,000 reviews in 1-2 seconds for fast iteration
- **Full Dataset** — Takes longer but loads all reviews for complete analysis
- **Smart Data Loading** — Only loads essential columns (rating, text, title, helpful_vote)
- **Cached Processing** — Streamlit caches data so reloads are instant

## Dataset

Uses Amazon Arts, Crafts & Sewing product reviews (JSONL format). Each record includes:
- `rating` — Star rating (1-5)
- `text` — Review text
- `title` — Review title
- `helpful_vote` — Number of helpful votes
