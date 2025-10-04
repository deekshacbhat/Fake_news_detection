# # pub_c90e37160e37479281e8f8fa42d9a3d7
# # -*- coding: utf-8 -*-
# """
# Fake News Detection Project using NewsData.io
# - Fetches Indian news in English
# - Accepts user news input
# - Checks if news is true/fake based on semantic similarity
# - Uses SVM + Naive Bayes hybrid model
# - Provides SHAP explainability
# """

# import requests
# import pandas as pd
# from googletrans import Translator
# from sentence_transformers import SentenceTransformer, util
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import shap
# import numpy as np

# # -------------------------------
# # 1. API Key & Fetching News
# # -------------------------------
# API_KEY = "pub_c90e37160e37479281e8f8fa42d9a3d7"

# def fetch_news(pages=2, country="in", language="en"):
#     news_list = []
#     for page in range(1, pages+1):
#         url = f"https://newsdata.io/api/1/latest?apikey=pub_c90e37160e37479281e8f8fa42d9a3d7&q=india%20news"
#         try:
#             r = requests.get(url)
#             data = r.json()
#             for item in data.get('results', []):
#                 news_list.append({
#                     "title": item.get('title', ''),
#                     "description": item.get('description', ''),
#                     "date": item.get('pubDate', ''),
#                     "source": item.get('source_id', ''),
#                 })
#         except Exception as e:
#             print(f"Error fetching page {page}: {e}")
#     df = pd.DataFrame(news_list)
#     return df

# # -------------------------------
# # 2. Translation (if needed)
# # -------------------------------
# translator = Translator()
# def translate_to_english(text):
#     try:
#         return translator.translate(text, dest='en').text
#     except:
#         return text

# # -------------------------------
# # 3. Semantic Similarity
# # -------------------------------
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def check_similarity(input_news, news_df, threshold=0.7):
#     input_news = translate_to_english(input_news)
#     input_emb = model.encode(input_news, convert_to_tensor=True)

#     similarities = []
#     for idx, row in news_df.iterrows():
#         news_text = row['title'] + " " + row['description']
#         news_text = translate_to_english(news_text)
#         news_emb = model.encode(news_text, convert_to_tensor=True)
#         sim_score = util.cos_sim(input_emb, news_emb).item()
#         similarities.append(sim_score)

#     news_df['similarity'] = similarities
#     max_sim = news_df['similarity'].max() if len(similarities) > 0 else 0
#     closest_news = news_df.loc[news_df['similarity'].idxmax()] if len(news_df) > 0 else None
#     is_real = max_sim >= threshold

#     return is_real, closest_news, max_sim

# # -------------------------------
# # 4. Hybrid Model: SVM + Naive Bayes
# # -------------------------------
# def train_hybrid_model(news_df):
#     news_df['label'] = news_df['similarity'].apply(lambda x: 1 if x > 0.7 else 0)
#     X = np.array(news_df['similarity']).reshape(-1, 1)
#     y = news_df['label']
#     le = LabelEncoder()
#     y = le.fit_transform(y)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     svm_model = SVC(probability=True)
#     nb_model = GaussianNB()

#     svm_model.fit(X_train, y_train)
#     nb_model.fit(X_train, y_train)

#     return svm_model, nb_model, le

# # -------------------------------
# # 5. SHAP Explainability
# # -------------------------------
# def explain_prediction(model, similarity_score):
#     explainer = shap.KernelExplainer(model.predict_proba, np.array([[0.5]]))
#     shap_values = explainer.shap_values(np.array([[similarity_score]]))
#     return shap_values

# # -------------------------------
# # 6. Main Program
# # -------------------------------
# if __name__ == "__main__":
#     print("Fetching latest Indian news...")
#     news_df = fetch_news(pages=3)
#     print("Fetched News Count:", len(news_df))

#     if len(news_df) == 0:
#         print("No news fetched. Check API key or NewsData.io source.")
#     else:
#         input_news = input("Enter news to verify: ")

#         # Semantic similarity check
#         is_real, closest_news, score = check_similarity(input_news, news_df)
#         print("\nSemantic Similarity Check:")
#         print("Likely Real" if is_real else "Likely Fake", f"(Similarity={score:.2f})")
#         if closest_news is not None:
#             print("Closest news from source:", closest_news['title'])
#             print("Published on:", closest_news['date'])
#             print("Source:", closest_news['source'])

#         # Train hybrid model
#         svm_model, nb_model, le = train_hybrid_model(news_df)

#         # Predict using hybrid
#         svm_pred = svm_model.predict(np.array([[score]]))
#         nb_pred = nb_model.predict(np.array([[score]]))
#         print("\nHybrid Model Predictions:")
#         print("SVM:", "Real" if svm_pred[0]==1 else "Fake")
#         print("Naive Bayes:", "Real" if nb_pred[0]==1 else "Fake")

#         # Explain SVM prediction
#         print("\nSHAP Explainability for SVM:")
#         shap_values = explain_prediction(svm_model, score)
#         print("SHAP Values (Feature importance):", shap_values)

""" 
Fake News Detection Project using NewsData.io 
- Fetches Indian news in English 
- Accepts user news input 
- Checks if news is true/fake based on semantic similarity 
- Uses SVM + Naive Bayes hybrid model (similarity + TF-IDF) 
- Provides SHAP explainability 
"""

# import requests
# import pandas as pd
# from googletrans import Translator
# from sentence_transformers import SentenceTransformer, util
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# import shap
# import numpy as np

# # ------------------------------- #
# # 1. API Key & Fetching News
# # ------------------------------- #
# API_KEY = "pub_c90e37160e37479281e8f8fa42d9a3d7"

# def fetch_news(pages=2, country="in", language="en"):
#     news_list = []
#     for page in range(1, pages+1):
#         url = f"https://newsdata.io/api/1/latest?apikey=pub_c90e37160e37479281e8f8fa42d9a3d7&q=indian%20news"
#         try:
#             r = requests.get(url)
#             data = r.json()
#             for item in data.get('results', []):
#                 news_list.append({
#                     "title": item.get('title', ''),
#                     "description": item.get('description', ''),
#                     "date": item.get('pubDate', ''),
#                     "source": item.get('source_id', ''),
#                 })
#         except Exception as e:
#             print(f"Error fetching page {page}: {e}")
#     df = pd.DataFrame(news_list)
#     return df

# # ------------------------------- #
# # 2. Translation (if needed)
# # ------------------------------- #
# translator = Translator()
# def translate_to_english(text):
#     try:
#         return translator.translate(text, dest='en').text
#     except:
#         return text

# # ------------------------------- #
# # 3. Semantic Similarity
# # ------------------------------- #
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def check_similarity(input_news, news_df, threshold=0.7):
#     input_news = translate_to_english(input_news).strip().lower()

#     similarities = []
#     for idx, row in news_df.iterrows():
#         news_text = (row['title'] + " " + row['description']).strip().lower()
#         news_text = translate_to_english(news_text)

#         # ✅ Exact match check
#         if input_news == news_text:
#             sim_score = 1.0
#         else:
#             # Semantic similarity
#             input_emb = model.encode(input_news, convert_to_tensor=True)
#             news_emb = model.encode(news_text, convert_to_tensor=True)
#             sim_score = util.cos_sim(input_emb, news_emb).item()

#         similarities.append(sim_score)

#     news_df['similarity'] = similarities
#     max_sim = news_df['similarity'].max() if len(similarities) > 0 else 0
#     closest_news = news_df.loc[news_df['similarity'].idxmax()] if len(news_df) > 0 else None
#     is_real = max_sim >= threshold

#     return is_real, closest_news, max_sim


# # ------------------------------- #
# # 4. Hybrid Model: SVM + Naive Bayes
# # ------------------------------- #
# def train_hybrid_model(news_df):
#     # Label by similarity baseline
#     news_df['label'] = news_df['similarity'].apply(lambda x: 1 if x > 0.7 else 0)

#     # TF-IDF features from text
#     texts = (news_df['title'].fillna('') + " " + news_df['description'].fillna('')).tolist()
#     vectorizer = TfidfVectorizer(max_features=300)
#     tfidf_features = vectorizer.fit_transform(texts).toarray()

#     # Add similarity score as extra feature
#     X = np.hstack([tfidf_features, news_df['similarity'].values.reshape(-1,1)])
#     y = news_df['label'].values

#     le = LabelEncoder()
#     y = le.fit_transform(y)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     svm_model = SVC(probability=True)
#     nb_model = GaussianNB()

#     svm_model.fit(X_train, y_train)
#     nb_model.fit(X_train, y_train)

#     return svm_model, nb_model, vectorizer, le

# # ------------------------------- #
# # 5. SHAP Explainability
# # ------------------------------- #
# def explain_prediction(model, features):
#     explainer = shap.KernelExplainer(model.predict_proba, np.zeros((1, features.shape[1])))
#     shap_values = explainer.shap_values(features)
#     return shap_values

# # ------------------------------- #
# # 6. Main Program
# # ------------------------------- #
# if __name__ == "__main__":
#     print("Fetching latest Indian news...")
#     news_df = fetch_news(pages=3)
#     print("Fetched News Count:", len(news_df))

#     if len(news_df) == 0:
#         print("No news fetched. Check API key or NewsData.io source.")
#     else:
#         input_news = input("Enter news to verify: ")

#         # Semantic similarity check
#         is_real, closest_news, score = check_similarity(input_news, news_df)
#         print("\nSemantic Similarity Check:")
#         print("Likely Real" if is_real else "Likely Fake", f"(Similarity={score:.2f})")
#         if closest_news is not None:
#             print("Closest news from source:", closest_news['title'])
#             print("Published on:", closest_news['date'])
#             print("Source:", closest_news['source'])

#         # Train hybrid model
#         svm_model, nb_model, vectorizer, le = train_hybrid_model(news_df)

#         # Prepare input for prediction
#         input_text = translate_to_english(input_news)
#         input_tfidf = vectorizer.transform([input_text]).toarray()
#         input_features = np.hstack([input_tfidf, [[score]]])

#         # Predict using hybrid
#         svm_pred = svm_model.predict(input_features)
#         nb_pred = nb_model.predict(input_features)

#         print("\nHybrid Model Predictions:")
#         print("SVM:", "Real" if svm_pred[0]==1 else "Fake")
#         print("Naive Bayes:", "Real" if nb_pred[0]==1 else "Fake")

#         # Explain SVM prediction
#         print("\nSHAP Explainability for SVM:")
#         shap_values = explain_prediction(svm_model, input_features)
#         print("SHAP Values (Feature importance):", shap_values)
#---------------------------------------------------------------------
import requests
import pandas as pd
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import spacy
import re
import shap
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ------------------------------- #
# 1. Translation + Utilities
# ------------------------------- #
translator = Translator()
model = SentenceTransformer('all-MiniLM-L6-v2')

def translate_to_english(text):
    """
    Converts input to English safely.
    Handles None or empty strings without throwing errors.
    """
    if not text:
        return ""
    try:
        detection = translator.detect(str(text))
        detected_lang = detection.lang
        if detected_lang != 'en':
            return translator.translate(str(text), src=detected_lang, dest='en').text
        return str(text)
    except:
        return str(text)


def extract_locations(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents if ent.label_ == "GPE"]

def location_match(text1, text2):
    loc1 = set(extract_locations(text1))
    loc2 = set(extract_locations(text2))
    if not loc1 or not loc2:
        return 1
    return int(bool(loc1 & loc2))

# hybrid decision (unchanged logic)
def hybrid_decision(svm_pred, nb_pred, svm_acc, nb_acc):
    if svm_pred == nb_pred:
        return svm_pred
    return svm_pred if svm_acc > nb_acc else nb_pred

# improved explain_prediction: prints TF-IDF top words found earlier
def explain_prediction_with_tfidf_and_shap(shap_vals, feature_names, tfidf_vector, top_n=5):
    """
    shap_vals: 1D array of SHAP contributions aligned with feature_names
    feature_names: list of names matching shap_vals order
    tfidf_vector: the TF-IDF vector for the input (array)
    """
    # Only show top_n influential features by absolute SHAP value
    abs_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    top_items = []
    for i in abs_idx:
        name = feature_names[i]
        contrib = shap_vals[i]
        # Skip trivial numeric features if not meaningful
        top_items.append((name, float(contrib)))
    return top_items

# existing TF-IDF highlighting function (keeps your original behavior but made robust)
def explain_prediction(features, feature_names, top_n=5):
    # features expected shape (1, n_features)
    row = np.asarray(features).reshape(-1)
    tfidf_features = row[:-2]
    similarity_score = row[-2]
    loc_match = row[-1]
    top_idx = np.argsort(tfidf_features)[::-1][:top_n]
    print("\nKey factors behind this prediction (TF-IDF based):")
    count = 0
    for i in top_idx:
        if tfidf_features[i] > 0 and not re.fullmatch(r"[0-9]+[a-zA-Z]*", feature_names[i]):
            print(f" - {feature_names[i]} (TF-IDF={tfidf_features[i]:.3f})")
            count += 1
            if count >= top_n:
                break
    print(f" - similarity_score ({similarity_score:.3f})")
    print(f" - location_match ({loc_match})")

# ------------------------------- #
# 2. Fetch News
# ------------------------------- #
# ------------------------------- #
# 2. Fetch News (updated for multilingual support)
# ------------------------------- #
# ------------------------------- #
# 2. Fetch News (simple multilingual translation)
# ------------------------------- #
def fetch_news(pages=2):
    news_list = []
    for page in range(1, pages + 1):
        url = f"https://newsdata.io/api/1/latest?apikey=pub_c90e37160e37479281e8f8fa42d9a3d7&q=kannada%20news"
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            for item in data.get('results', []):
                title = str(item.get('title') or "")
                desc = str(item.get('description') or "")

                # Translate fetched news to English safely
                try:
                    title_en = translate_to_english(title) if title.strip() else ""
                except Exception as e:
                    title_en = title

                try:
                    desc_en = translate_to_english(desc) if desc.strip() else ""
                except Exception as e:
                    desc_en = desc

                news_list.append({
                    "title": title_en,
                    "description": desc_en,
                    "date": item.get('pubDate', ''),
                    "source": item.get('source_id', ''),
                })
        except Exception as e:
            print(f"Error fetching page {page}: {e}")

    return pd.DataFrame(news_list)


# ------------------------------- #
# 3. Compute Semantic Similarity
# ------------------------------- #
def compute_similarity(input_news, news_df):
    input_trans = translate_to_english(input_news).strip().lower()
    input_emb = model.encode(input_trans, convert_to_tensor=True)
    news_texts = (news_df['title'].fillna('') + " " + news_df['description'].fillna('')).tolist()
    news_embs = model.encode(news_texts, convert_to_tensor=True)
    sims = util.cos_sim(input_emb, news_embs)[0].cpu().numpy()
    loc_matches = [location_match(input_trans, t) for t in news_texts]
    # ensure we don't modify original df unexpectedly: make a copy
    tmp = news_df.copy()
    tmp['similarity'] = sims
    tmp['location_match'] = loc_matches
    max_sim = float(np.max(sims)) if len(sims) else 0.0
    closest_news = tmp.loc[tmp['similarity'].idxmax()] if len(tmp) > 0 else None
    return max_sim, closest_news, input_trans

# ------------------------------- #
# 4. Train Models
# ------------------------------- #
def train_models(news_df):
    # IMPORTANT: keep your similarity-based detection but choose a slightly stricter labeling to reduce leakiness.
    # Label positive when similarity > 0.5 (you can tune). This keeps SVM/NB meaningful but still matches your logic.
    news_df = news_df.copy()
    news_df['label'] = news_df['similarity'].apply(lambda x: 1 if x > 0.5 else 0)

    # if only one class present in labels, return None models (keeps your fallback)
    if len(news_df['label'].unique()) < 2:
        return None, None, None, None, None, None

    texts = (news_df['title'].fillna('') + " " + news_df['description'].fillna('')).tolist()
    # Keep max_features same as before
    vectorizer = TfidfVectorizer(max_features=300)
    tfidf_features = vectorizer.fit_transform(texts).toarray()

    # Build combined features (TF-IDF + similarity + location)
    X = np.hstack([tfidf_features,
                   news_df['similarity'].values.reshape(-1,1),
                   news_df['location_match'].values.reshape(-1,1)])
    y = LabelEncoder().fit_transform(news_df['label'].values)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM and NB (kept)
    svm_model = SVC(probability=True).fit(X_train, y_train)
    nb_model = GaussianNB().fit(X_train, y_train)

    svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

    # Return training data too (useful for SHAP background)
    return svm_model, nb_model, vectorizer, (X_train, y_train), svm_acc, nb_acc

# ------------------------------- #
# 5. Main Program
# ------------------------------- #
if __name__ == "__main__":
    print("Fetching latest Indian news...")
    news_df = fetch_news(pages=3)
    print("Fetched News Count:", len(news_df))

    if len(news_df) == 0:
        print("No news fetched.")
    else:
        input_news = input("Enter news to verify: ")

        # Step 3: Compute similarity
        similarity_score, closest_news, input_trans = compute_similarity(input_news, news_df)

        # Step 4: Train models
                # Step 4: Prepare similarity & location features for training
        # (compute similarity between each article and your input news)
        input_emb = model.encode(input_trans, convert_to_tensor=True)
        news_texts = (news_df['title'].fillna('') + " " + news_df['description'].fillna('')).tolist()
        news_embs = model.encode(news_texts, convert_to_tensor=True)
        sims = util.cos_sim(input_emb, news_embs)[0].cpu().numpy()
        loc_matches = [location_match(input_trans, t) for t in news_texts]
        news_df['similarity'] = sims
        news_df['location_match'] = loc_matches

        # Step 5: Train models
        svm_model, nb_model, vectorizer, train_data, svm_acc, nb_acc = train_models(news_df)


        # Step 5: Prepare input features
        if closest_news is not None:
            closest_text = (closest_news['title'] or '') + " " + (closest_news['description'] or '')
            loc_feat = location_match(input_trans, closest_text)
        else:
            loc_feat = 1

        # Exact-copy or extremely-close override: if input is nearly identical to an existing article -> mark as real immediately
        # This addresses the issue you reported where exact-copy was classified as Fake.
        if similarity_score >= 0.95:
            final_label = 1
            decision_reason = "Exact or near-exact match to a fetched article (similarity >= 0.95). Overriding to Real."
            used_model_name = "Exact-match override"
            shap_top_words = []
            # build some prints below
        else:
            # Build TF-IDF for input using the trained vectorizer (or fallback)
            if vectorizer is not None:
                input_tfidf = vectorizer.transform([input_trans]).toarray()  # shape (1, n_tfidf)
                # Ensure shapes align: stack similarity and location as column vectors
                input_features = np.hstack([input_tfidf, np.array([[similarity_score, loc_feat]])])
            else:
                # Fallback (rare): build a vectorizer on just input
                fallback_vectorizer = TfidfVectorizer(max_features=300)
                input_tfidf = fallback_vectorizer.fit_transform([input_trans]).toarray()
                input_features = np.hstack([input_tfidf, np.array([[similarity_score, loc_feat]])])

            # Step 6: Predictions
            if svm_model is not None and nb_model is not None:
                svm_pred = int(svm_model.predict(input_features)[0])
                nb_pred = int(nb_model.predict(input_features)[0])
                final_label = hybrid_decision(svm_pred, nb_pred, svm_acc, nb_acc)
                # Determine which model's SHAP to show: if both agree, prefer that model; if disagree choose higher-accuracy model
                if svm_pred == nb_pred:
                    used_model = svm_model if svm_acc >= nb_acc else nb_model
                    used_model_name = "SVM" if svm_acc >= nb_acc else "NaiveBayes"
                else:
                    used_model = svm_model if svm_acc > nb_acc else nb_model
                    used_model_name = "SVM" if svm_acc > nb_acc else "NaiveBayes"
                decision_reason = f"Hybrid decision based on SVM({svm_acc:.3f}) and NB({nb_acc:.3f}); used: {used_model_name}."
            else:
                # fallback rule when models are None (single-class case)
                final_label = 1 if (similarity_score > 0.4 and loc_feat==1) else 0
                used_model = None
                used_model_name = "Fallback-rule"
                decision_reason = "Fallback rule used because models couldn't be trained (single-class labels)."

            # Step 7: SHAP explanation for chosen model (if available)
            shap_top_words = []
            try:
                if 'used_model' in locals() and used_model is not None and train_data is not None:
                    X_train, y_train = train_data
                    # select a small background (max 50 rows) to keep Kernel SHAP reasonable
                    bg = X_train[np.random.choice(X_train.shape[0], min(50, X_train.shape[0]), replace=False)]
                    # create a prediction function returning probability for class 1
                    def model_predict_proba(X_in):
                        # SVM and NB have predict_proba
                        probs = used_model.predict_proba(X_in)
                        # return prob of positive class (1)
                        if probs.ndim == 2:
                            return probs[:, 1]
                        else:
                            # defensive
                            return probs
                    # KernelExplainer expects a function that returns 2D array for each sample; wrap
                    explainer = shap.KernelExplainer(model_predict_proba, bg, link="identity")
                    # compute shap values for our single input (may take seconds)
                    shap_values = explainer.shap_values(input_features, nsamples=100)
                    # KernelExplainer returns array shape (1, n) for binary probability function
                    # make sure we have a 1D array of shap contributions:
                    if isinstance(shap_values, list):
                        shap_arr = np.array(shap_values)[0].reshape(-1)
                    else:
                        shap_arr = np.array(shap_values).reshape(-1)
                    # Build feature names
                    feature_names = (vectorizer.get_feature_names_out().tolist() if vectorizer else fallback_vectorizer.get_feature_names_out().tolist()) + ["similarity_score", "location_match"]
                    # Get top 8 words by absolute shap among TF-IDF features only
                    top_items = explain_prediction_with_tfidf_and_shap(shap_arr, feature_names, input_tfidf, top_n=8)
                    # Filter to only words that exist in TF-IDF vocabulary (exclude similarity/location)
                    shap_top_words = [(n, v) for (n, v) in top_items if n not in ("similarity_score", "location_match")]
                else:
                    shap_top_words = []
            except Exception as ex:
                # SHAP can fail on some environments or take long — in that case, skip gracefully
                shap_top_words = []
                print("Warning: SHAP explanation failed or was skipped:", ex)

        # Step 8: Show similarity results
        print("\nSemantic Similarity Check:")
        print("Similarity Score:", f"{similarity_score:.2f}")
        if closest_news is not None:
            print("Closest news from source:", closest_news['title'])
            print("Published on:", closest_news['date'])
            print("Source:", closest_news['source'])

        # Step 9: Hybrid Model output
        print("\nHybrid Model Prediction:", "Real" if final_label==1 else "Fake")
        print("Decision detail:", decision_reason)

        # Step 10: Explainable AI (TF-IDF based)
        # Build feature list (for TF-IDF style explain)
        if vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out().tolist() + ["similarity_score", "location_match"]
        else:
            feature_names = fallback_vectorizer.get_feature_names_out().tolist() + ["similarity_score", "location_match"]

        # Recreate input_features array for explain_prediction call if not in exact-override path
        if similarity_score >= 0.95:
            # build a best-effort TF-IDF display for the user
            try:
                # Try to show important TF-IDF tokens (fallback): use vectorizer if available
                if vectorizer is not None:
                    input_tfidf_display = vectorizer.transform([input_trans]).toarray()
                else:
                    input_tfidf_display = fallback_vectorizer.transform([input_trans]).toarray()
                features_for_explain = np.hstack([input_tfidf_display, np.array([[similarity_score, loc_feat]])])
            except:
                features_for_explain = np.hstack([np.zeros(len(feature_names)-2), np.array([similarity_score, loc_feat])])
            explain_prediction(features_for_explain, feature_names, top_n=5)
        else:
            explain_prediction(input_features, feature_names, top_n=5)

        # SHAP top words display (friendly)
        if len(shap_top_words) > 0:
            print("\nTop words influencing this prediction (SHAP):")
            for w, contrib in shap_top_words[:8]:
                sign = "+" if contrib > 0 else "-"
                print(f" - {w}: {sign}{abs(contrib):.4f}")
        else:
            print("\n(SHAP explanation not available or skipped.)")
def main_prediction_output(input_text,langs=None):
    """
    Wrapper so Flask (app.py) can call the same logic.
    It mimics your __main__ flow but returns the output instead of printing only.
    """
    print("Fetching latest Indian news...")
    news_df = fetch_news(pages=3)
    print("Fetched News Count:", len(news_df))

    if len(news_df) == 0:
        return {"error": "No news fetched."}

    # Compute similarity
    similarity_score, closest_news, input_trans = compute_similarity(input_text, news_df)

    # Prepare features and train models
    input_emb = model.encode(input_trans, convert_to_tensor=True)
    news_texts = (news_df['title'].fillna('') + " " + news_df['description'].fillna('')).tolist()
    news_embs = model.encode(news_texts, convert_to_tensor=True)
    sims = util.cos_sim(input_emb, news_embs)[0].cpu().numpy()
    loc_matches = [location_match(input_trans, t) for t in news_texts]
    news_df['similarity'] = sims
    news_df['location_match'] = loc_matches

    svm_model, nb_model, vectorizer, train_data, svm_acc, nb_acc = train_models(news_df)

    # Fallback or hybrid logic reuse (same as yours)
    if closest_news is not None:
        closest_text = (closest_news['title'] or '') + " " + (closest_news['description'] or '')
        loc_feat = location_match(input_trans, closest_text)
    else:
        loc_feat = 1

    if similarity_score >= 0.95:
        final_label = 1
        decision_reason = "Exact or near-exact match (similarity >= 0.95). Overriding to Real."
    else:
        if svm_model is not None and nb_model is not None:
            input_tfidf = vectorizer.transform([input_trans]).toarray()
            input_features = np.hstack([input_tfidf, np.array([[similarity_score, loc_feat]])])
            svm_pred = int(svm_model.predict(input_features)[0])
            nb_pred = int(nb_model.predict(input_features)[0])
            final_label = hybrid_decision(svm_pred, nb_pred, svm_acc, nb_acc)
            decision_reason = f"Hybrid decision (SVM={svm_acc:.3f}, NB={nb_acc:.3f})"
        else:
            final_label = 1 if (similarity_score > 0.4 and loc_feat == 1) else 0
            decision_reason = "Fallback rule used because models couldn't be trained (single-class labels)."

    return {
        "prediction": "Real" if final_label == 1 else "Fake",
        "similarity_score": round(similarity_score, 3),
        "closest_news": closest_news.to_dict() if closest_news is not None else None,
        "decision_detail": decision_reason
    }
