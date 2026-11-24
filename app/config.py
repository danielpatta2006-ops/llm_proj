import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv('DB_PATH', str(BASE_DIR / 'interview_prep.db'))
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
ENV_API_KEY = os.getenv('GROQ_API_KEY', '')
JOB_ROLES = ['Software Developer','Data Analyst','AI Engineer','Web Developer','Data Scientist','Cloud/DevOps Engineer','Cybersecurity Analyst']
CATEGORIES = ['Technical','HR','Mixed']
DIFFICULTIES = ['Beginner','Intermediate','Advanced']
WEIGHTS = {'technical_accuracy':.30,'relevance':.20,'completeness':.20,'clarity':.15,'communication':.15}
