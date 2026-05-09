from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from orchestration.state import graphState
from orchestration.graph import AI_researcher, Summary_generator, Information_retrieval
import re, os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
load_dotenv()

passw=os.getenv("DB_PASSWORD")
hf_api=os.getenv("hf_api")
conn=psycopg2.connect(

    host="research-project-akshatavyas01-cf54.e.aivencloud.com",
    port=27517,
    database="defaultdb",
    user="avnadmin",
    password=passw,
    sslmode="require"
)
register_vector(conn)

cur=conn.cursor()

model=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=hf_api
)


app=FastAPI()

async def DB_cache(topic:str, task=False):
    try:
        result_list=[]
        topic_embedded=model.embed_query(topic)
        cur.execute("SELECT *,  (1 - (embedding <=> %s::vector)) AS similarity FROM research_cache  WHERE (1 -(embedding <=> %s::vector)) > 0.80 ORDER BY embedding <=> %s::vector LIMIT 4",(topic_embedded,topic_embedded,topic_embedded))
        rows=cur.fetchall()
        if rows:
            if task:
                for row in rows:
                    summary=row[3]
                    if summary:
                        result_list.append(summary)
            else:
                for row in rows:
                    summary=row[3]
                    report=row[4]
                    feedback=row[5]
                    if summary and report and feedback:
                        result_list.append({"Summary":summary,"Report":report, "Feedback":feedback}) 
                
                return result_list
        else:
            return "NOT FOUND"
    except Exception as e:
        conn.rollback() 
        return {"error":str(e)}


async def run_graph(graph, topic:str, key:str):
    try:
        result= await graph.ainvoke(graphState(topic=topic))
        data=result.get(key)

        if isinstance (data,str):
            data=data.replace("\\n","\n")
        
        return data
    except Exception as e:
        return {"error":str(e)}
    
async def Unique_research(topic:str):
    topic_embedded=model.embed_query(topic)
    result=await AI_researcher.ainvoke(graphState(topic=topic))
    result_report=result.get('report')
    if result_report:
        result_report = re.sub(r"(\\n|\n|\|)", " ", result_report)

    summary=result.get("summary")
    feedback=result.get("feedback")
    cur.execute('''INSERT INTO research_cache (topic, embedding, summary, report, feedback) VALUES (%s, %s, %s, %s, %s)''',(topic, topic_embedded,summary,result_report,feedback))
    conn.commit()
    return {
        "Summary":summary or "ERROR",
        "Report":result_report or "ERROR",
        "Feedback":feedback or "ERROR"
        }


@app.get("/reseacher/report_generator")
async def report_generator(topic:str, resubmit:bool):
    try:
        if resubmit:
            result=await Unique_research(topic)
            return result
        cache= await DB_cache(topic)
        if cache=="NOT FOUND":
            result=await Unique_research(topic)
            return result
        else:
            return cache
    except Exception as e:
        return {'ERROR':str(e)}
    
@app.get("/reseacher/summary_generator")
async def summary_generator(topic:str, resubmit:bool):
    try:
        if resubmit:
            summary = await run_graph(Summary_generator,topic,"summary")
            return {"Summary":summary or "ERROR"}
        cache= await DB_cache(topic,True)
        if cache=="NOT FOUND":
            summary = await run_graph(Summary_generator,topic,"summary")
            return {"Summary":summary or "ERROR"}
        else:
            result=cache
            return {"Summary":result}
    except Exception as e:
        return {'ERROR':str(e)}

    
@app.get("/reseacher/information_retrival")
async def Information_generator(topic:str):
    info = await run_graph(Information_retrieval,topic,"researched_info")
    return {"Information":info or "ERROR"}

