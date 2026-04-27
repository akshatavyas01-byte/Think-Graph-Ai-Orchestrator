from fastapi import FastAPI
from .state import graphState
from .graph import AI_researcher, Summary_generator, Information_retrieval


app=FastAPI()


@app.get("/reseacher/report_generator")
async def report_generator(topic:str):
    state=graphState(
            topic=topic
        )
    try:
        result=await AI_researcher.ainvoke(state)
        result_report=result.get('report')
        dict={'Summary':result.get('summary'), 'Report':result_report.replace("\\n", "\n") if result_report is not None else 'ERROR','Feedback':result.get('feedback')}
        return dict
    except Exception as e:
        return {'ERROR':e}
    
@app.get("/reseacher/summary_generator")
async def summary_generator(topic:str):
    state=graphState(
            topic=topic
        )
    try:
        result=await Summary_generator.ainvoke(state)
        dict={'Summary':result.get('summary')}
        return dict
    except Exception as e:
        return {'ERROR':e}

    
@app.get("/reseacher/information_retrival")
async def Information_generator(topic:str):
    state=graphState(
            topic=topic
        )
    try:
        result=await Information_retrieval.ainvoke(state)
        info=result.get('researched_info')
        dict={'Information_retrieved':info.replace("\\n", "\n") if info is not None else 'ERROR'}
        return dict
    except Exception as e:
        return {'ERROR':e}



   
    
    


