import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Load model and encoder ---
pipeline = joblib.load('atis_model.pkl')
le = joblib.load('atis_label_encoder.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"'", '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words]
    return ' '.join(words)


# --- Intent descriptions ---
intent_descriptions = {
    'atis_flight': '✈️ Flight Inquiry — searching for flights',
    'atis_airfare': '💰 Airfare Inquiry — asking about prices or fares',
    'atis_ground_service': '🚌 Ground Service — transport from airport',
    'atis_airline': '🏢 Airline Inquiry — asking about an airline',
    'atis_abbreviation': '🔤 Abbreviation — asking what a code means',
    'atis_aircraft': '🛩️ Aircraft — asking about plane type',
    'atis_flight_time': '🕐 Flight Time — asking about schedules',
    'atis_quantity': '🔢 Quantity — asking how many',
    'atis_other': '❓ Other — general or rare inquiry'
}

# --- Page config ---
st.set_page_config(
    page_title="Flight Intent Classifier",
    page_icon="✈️",
    layout="centered"
)

# --- Header ---
st.title("✈️ Flight Query Intent Classifier")
st.markdown("### Powered by NLP + Logistic Regression")
st.markdown(
    "Type any airline-related query and the model will "
    "instantly classify what you're asking about."
)
st.markdown("---")

# --- Input ---
st.subheader("💬 Enter Your Query")
user_input = st.text_input(
    "Type your flight query here:",
    placeholder="e.g. what is the cheapest flight from nairobi to london"
)

# --- Predict ---
if st.button("Classify Intent"):
    if user_input.strip() == "":
        st.warning("Please enter a query first.")
    else:
        cleaned = clean_text(user_input)
        prediction = pipeline.predict([cleaned])[0]
        intent = le.inverse_transform([prediction])[0]
        probability = pipeline.predict_proba([cleaned])[0].max()

        st.markdown("---")
        st.subheader("🎯 Predicted Intent")
        st.success(f"**{intent_descriptions.get(intent, intent)}**")

        st.metric("Confidence", f"{probability:.1%}")

        if probability < 0.5:
            st.warning(
                "⚠️ Low confidence — this query may be ambiguous "
                "or outside the model's training scope."
            )

        st.markdown("#### 🔍 How it was classified:")
        st.code(f"""
Original query:  {user_input}
Cleaned text:    {cleaned}
Predicted class: {intent}
Confidence:      {probability:.1%}
        """)

# --- Example queries ---
st.markdown("---")
st.subheader("💡 Try these examples")
examples = [
    "what is the cheapest flight from boston to london",
    "what airlines fly from nairobi to dubai",
    "how do i get from the airport to downtown",
    "what does the code UA mean",
    "what type of aircraft does delta use",
    "what time does the morning flight arrive"
]
for example in examples:
    st.markdown(f"- *{example}*")

# --- Footer ---
st.markdown("---")
st.caption(
    "Model: Logistic Regression + TF-IDF | "
    "Accuracy: 92% | Trained on ATIS dataset"
)
