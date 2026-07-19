"""
Sentiment Analysis Web App
---------------------------
Analyzes the sentiment of any text (Positive / Negative / Neutral)
with a confidence score — similar to what your resume project describes.

Built by Jaya Harish J
"""

import streamlit as st
import os
import joblib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .result-positive {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 20px; border-radius: 10px; margin-top: 15px;
    }
    .result-negative {
        background: #f8d7da; border-left: 5px solid #dc3545;
        padding: 20px; border-radius: 10px; margin-top: 15px;
    }
    .result-neutral {
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 20px; border-radius: 10px; margin-top: 15px;
    }
    .big-label { font-size: 1.8rem; font-weight: 700; }
    .conf-text { font-size: 1rem; color: #555; margin-top: 5px; }
    .history-item {
        background: white; border-radius: 8px; padding: 10px 15px;
        margin-bottom: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        display: flex; justify-content: space-between; align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Train the model (runs once, cached so it stays fast) ─────────
@st.cache_resource
def train_model():
    """
    Trains a TF-IDF + Logistic Regression sentiment classifier.
    Uses a built-in dataset of ~120 labeled reviews so the app
    works standalone without any uploaded files.
    Cached by Streamlit so it only trains ONCE per session.
    """
    positives = [
        "This product is absolutely amazing, I love it!",
        "Fantastic experience, highly recommended to everyone.",
        "Great quality and fast delivery, very happy.",
        "Excellent service, the team was very helpful.",
        "I'm so happy with this purchase, works perfectly.",
        "Outstanding performance, exceeded all my expectations.",
        "Wonderful product, will definitely buy again.",
        "Very satisfied with the quality and price.",
        "Best purchase I have made this year, superb!",
        "Loved it! Packaging was nice and product is top-notch.",
        "Really impressed with the speed and accuracy.",
        "The customer support was incredibly responsive and kind.",
        "Five stars! Perfect in every way.",
        "Smooth experience from start to finish.",
        "Highly efficient, saved me so much time.",
        "Beautiful design and easy to use.",
        "Exactly what I was looking for, great value.",
        "The app works flawlessly, no bugs at all.",
        "Delivered on time and in perfect condition.",
        "Very user-friendly and intuitive interface.",
        "Exceptional quality for the price paid.",
        "I recommend this to all my friends and family.",
        "Superb craftsmanship and attention to detail.",
        "Works like a charm, very impressed.",
        "The results were beyond what I expected.",
        "Incredible value for money, very happy.",
        "Awesome product, looks and feels premium.",
        "Quick response and very professional team.",
        "Completely satisfied with the overall experience.",
        "This is the best app I have used in a long time.",
        "Great app, very clean and easy to navigate.",
        "Loved every feature, nothing to complain about.",
        "Outstanding quality, packaging was also great.",
        "Very reliable and consistent performance.",
        "Delightful experience, will come back again.",
        "Brilliant product, works perfectly every time.",
        "The interface is clean and very intuitive.",
        "Exceeded my expectations in every aspect.",
        "Very good product, fast delivery and great packaging.",
        "Absolutely worth every rupee spent.",
    ]
    negatives = [
        "This product is terrible, complete waste of money.",
        "Very disappointed with the quality, not as described.",
        "Worst experience ever, will not buy again.",
        "The product broke after just two days of use.",
        "Horrible customer service, nobody responded to my issue.",
        "Completely useless, does not work at all.",
        "Very poor quality, totally not worth the price.",
        "Terrible packaging, the item arrived damaged.",
        "Do not buy this, it is a total scam.",
        "Awful product, nothing like the pictures shown.",
        "Disgusting experience, will never order again.",
        "The delivery was extremely late and item was wrong.",
        "Stopped working after one week, very poor build.",
        "I want a refund immediately, this is unacceptable.",
        "Very bad quality control, received a defective item.",
        "The app keeps crashing every time I open it.",
        "Extremely slow and frustrating to use.",
        "Waste of time and money, deeply regret buying.",
        "The support team was rude and unhelpful.",
        "Instructions were unclear and product did not work.",
        "Returned it immediately, absolutely terrible.",
        "False advertising, completely different from what shown.",
        "Poor performance and very bad user experience.",
        "Not recommended at all, save your money.",
        "The battery drains in less than an hour.",
        "Terrible build quality, looks cheap and feels cheap.",
        "Broken on arrival, very disappointed with the brand.",
        "Slow shipping, poor packing, bad product overall.",
        "Regret this purchase completely.",
        "One of the worst products I have ever bought.",
        "Frustrating experience from beginning to end.",
        "Very laggy and unresponsive, complete disaster.",
        "Came with missing parts, very bad quality check.",
        "Extremely dissatisfied, will warn my friends.",
        "The product is faulty and the refund process is terrible.",
        "Keeps disconnecting every few minutes, useless.",
        "Waste of money, not worth even half the price.",
        "The smell is awful and the material is very low quality.",
        "Completely broken after first use.",
        "Terrible experience, would give zero stars if I could.",
    ]
    neutrals = [
        "The product is okay, nothing special.",
        "It works as expected, nothing more nothing less.",
        "Average quality, does the basic job.",
        "Decent product for the price, not amazing.",
        "Neither good nor bad, just okay.",
        "The delivery was on time, product is average.",
        "It does what it is supposed to do.",
        "Not great, not terrible, just normal.",
        "Acceptable quality but nothing impressive.",
        "The product is fine, met basic expectations.",
        "Average experience overall.",
        "Could be better but not bad either.",
        "It is what it is, nothing exceptional.",
        "Mediocre quality, expected more for the price.",
        "It functions correctly but has room for improvement.",
        "Standard product, no surprises.",
        "Works fine for basic use.",
        "Not bad, but also not great.",
        "Reasonable quality for the price.",
        "The experience was fairly ordinary.",
        "Somewhat useful but has limitations.",
        "It is satisfactory for everyday use.",
        "The item is acceptable.",
        "Not impressive but also not a problem.",
        "The product meets minimum expectations.",
        "It is okay, nothing to rave about.",
        "Passable quality overall.",
        "It gets the job done, nothing more.",
        "Fairly standard, as expected.",
        "Good enough for basic tasks.",
        "Ordinary product with no standout features.",
        "Adequate for the price point.",
        "Simple and functional.",
        "Nothing special but serves the purpose.",
        "Moderate satisfaction with the product.",
        "The quality is average at best.",
        "No major complaints but also no praise.",
        "Works fine but feels basic.",
        "Typical product experience.",
        "Meets expectations without exceeding them.",
    ]

    texts  = positives + negatives + neutrals
    labels = (["Positive"] * len(positives) +
              ["Negative"] * len(negatives) +
              ["Neutral"]  * len(neutrals))

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, C=2.0)),
    ])
    model.fit(texts, labels)
    return model

model = train_model()

# ── Example texts for quick demo ─────────────────────────────────
EXAMPLES = {
    "😊 Positive review":  "This product is absolutely amazing, works perfectly and arrived on time!",
    "😠 Negative review":  "Terrible quality, broke after two days. Complete waste of money.",
    "😐 Neutral review":   "The product is okay, does its job but nothing special.",
    "💬 Customer feedback":"The delivery was late but the item itself was decent quality.",
    "📧 Support email":    "I am not satisfied with the service I received last week.",
}

# ── Session state for history ─────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ──────────────────────────────────────────────────────
st.title("🧠 Sentiment Analysis App")
st.caption("Detects whether a text is Positive, Negative, or Neutral — Built by Jaya Harish J")
st.markdown("---")

# ── Main layout ──────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Analyze any text")

    # Quick example buttons
    st.write("**Quick examples:**")
    ex_cols = st.columns(len(EXAMPLES))
    selected_example = None
    for i, (label, text) in enumerate(EXAMPLES.items()):
        with ex_cols[i]:
            if st.button(label, use_container_width=True):
                selected_example = text

    user_text = st.text_area(
        "Type or paste any text below:",
        value=selected_example if selected_example else "",
        placeholder="e.g. This product is amazing and I love it!",
        height=140,
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            sentiment = model.predict([user_text])[0]
            proba     = model.predict_proba([user_text])[0]
            classes   = model.classes_
            conf      = max(proba)
            conf_dict = dict(zip(classes, proba))

            # Result card
            css_class = {
                "Positive": "result-positive",
                "Negative": "result-negative",
                "Neutral":  "result-neutral"
            }[sentiment]

            emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}[sentiment]

            st.markdown(f"""
            <div class='{css_class}'>
                <div class='big-label'>{emoji} {sentiment}</div>
                <div class='conf-text'>Confidence: {conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**Confidence breakdown:**")
            for label, score in sorted(conf_dict.items(), key=lambda x: -x[1]):
                st.progress(score, text=f"{label}: {score*100:.1f}%")

            # Save to history
            st.session_state.history.insert(0, {
                "text": user_text[:60] + ("..." if len(user_text) > 60 else ""),
                "sentiment": sentiment,
                "confidence": conf
            })

with right:
    st.subheader("📋 Analysis History")
    if not st.session_state.history:
        st.info("Your analyzed texts will appear here.")
    else:
        for i, item in enumerate(st.session_state.history[:10]):
            emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}.get(item["sentiment"], "❓")
            color = {"Positive": "#28a745", "Negative": "#dc3545", "Neutral": "#ffc107"}.get(item["sentiment"], "#888")
            st.markdown(f"""
            <div class='history-item'>
                <span style='flex:1; font-size:0.85rem; color:#333'>{item['text']}</span>
                <span style='color:{color}; font-weight:700; margin-left:10px'>{emoji} {item['sentiment']}</span>
                <span style='color:#aaa; font-size:0.8rem; margin-left:8px'>{item['confidence']*100:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

st.markdown("---")
st.caption("Sentiment Analysis App · TF-IDF + Logistic Regression · Built by Jaya Harish J")
