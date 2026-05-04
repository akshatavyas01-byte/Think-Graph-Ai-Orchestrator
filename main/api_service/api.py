from fastapi import FastAPI
from orchestration.state import graphState
from orchestration.graph import AI_researcher, Summary_generator, Information_retrieval
import re

app=FastAPI()

async def run_graph(graph, topic:str, key:str):
    try:
        result= await graph.ainvoke(graphState(topic=topic))
        data=result.get(key)

        if isinstance (data,str):
            data=data.replace("\\n","\n")
        
        return data
    except Exception as e:
        return {"error":str(e)}


@app.get("/reseacher/report_generator")
async def report_generator(topic:str):
    try:
        result=await AI_researcher.ainvoke(graphState(topic=topic))
        result_report=result.get('report')
       
        if result_report:
            result_report = re.sub(r"(\\n|\n|\|)", " ", result_report)

        return {
            "Summary":result.get("summary") or "ERROR",
            "Report":result_report or "ERROR",
            "Feedback":result.get("feedback") or "ERROR"
        }

    except Exception as e:
        return {'ERROR':str(e)}
    
@app.get("/reseacher/summary_generator")
async def summary_generator(topic:str):
    summary = await run_graph(Summary_generator,topic,"summary")
    return {"Summary":summary or "ERROR"}

    
@app.get("/reseacher/information_retrival")
async def Information_generator(topic:str):
    info = await run_graph(Information_retrieval,topic,"researched_info")
    return {"Information":info or "ERROR"}

