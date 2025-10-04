## Installation & Setup

1. Clone the repository:
git clone <your-repo-url>
cd <your-repo-folder>

2. Install all required Python packages:
pip install -r requirements.txt
(This will install Flask, scikit-learn, pandas, numpy, matplotlib, seaborn, sentence-transformers, spaCy, shap, googletrans, and more.)

3. Download the spaCy English language model:
python -m spacy download en_core_web_sm
(This is required for spaCy to process English text correctly.)

4. Run the Flask app (if your project has a Flask server):
 On Linux/macOS
export FLASK_APP=app.py
flask run

 On Windows (Command Prompt)
set FLASK_APP=app.py
flask run
(Replace app.py with your main Flask file if different.)

5. Open the app in your browser:
http://127.0.0.1:5000
(The app should now be running locally.)
