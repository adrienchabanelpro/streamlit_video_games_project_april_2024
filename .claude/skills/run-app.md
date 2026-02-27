# Run App Skill

Launch the Streamlit application locally.

## Command
```bash
cd /Users/mac_adrien/Projects/pro/streamlit_video_games_project_april_2024
streamlit run source/main.py
```

## Expected
- Opens at http://localhost:8501
- Sidebar with 9 navigation options
- No errors in terminal

## Troubleshooting
- If port 8501 is busy: `streamlit run source/main.py --server.port 8502`
- If module not found: `pip install -r requirements.txt`
- If NLTK data missing: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"`
