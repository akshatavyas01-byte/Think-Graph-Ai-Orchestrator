import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

password=os.getenv("DB_PASSWORD")
conn = psycopg2.connect(
    host="research-project-akshatavyas01-cf54.e.aivencloud.com",
    port=27517,
    database="defaultdb",
    user="avnadmin",
    password=password,
    sslmode="require"
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