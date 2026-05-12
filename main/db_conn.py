import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

Url=os.getenv("DB_URL")
conn = psycopg2.connect(
    Url
)

cur=conn.cursor()
cur.execute("CREATE EXTENSION VECTOR;")

cur.execute('''
CREATE TABLE research_cache (
    id SERIAL PRIMARY KEY,
    topic TEXT,
    embedding VECTOR(384),
    summary TEXT,
    report TEXT,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);''')

cur.fetchone()
conn.commit()

cur.close()
conn.close()