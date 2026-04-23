from state import graphState
# from langchain_community.tools import DuckDuckGoSearchResults 
from langchain_tavily import TavilySearch
from langchain_community.retrievers import WikipediaRetriever
import wikipedia
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain.messages import SystemMessage, HumanMessage
from langchain_classic.prompts import PromptTemplate
# from langchain_core.vectorstores import FAISS

from langchain_groq import ChatGroq
from pydantic import SecretStr
from dotenv import load_dotenv
import os

load_dotenv()

#LLM MODEL BLOCK
hf_api=os.getenv("hf_api")

groq_api=os.getenv("groq_api")


research_model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(groq_api) if groq_api else None,
    temperature=0.1 
)


llm=HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=hf_api,
    temperature=0.1,
    top_k=1,
    top_p= 0.9
)

summery_model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(groq_api) if groq_api else None,
    temperature=0.1 
)
 
report_model=ChatHuggingFace(llm=llm)


#RETRIVER & WEB SEARCH TOOL 

wiki_retriever=WikipediaRetriever(wiki_client=wikipedia, top_k_results=2)

websearch=TavilySearch( include_content=True, max_results=4)
#TOOLS FOR RESEARCH AGENT

@tool
def wikipedia_retriever_tool(topic:str)->str:
    '''Get information related from wikipedia database for research.
        This tool will return page content of the wikipedia documents.'''
    document=wiki_retriever.invoke(topic)
    text=''
    for doc in document:
        text+=doc.page_content 
    return text

@tool 
def websearch_tool(topic:str)->list:
    ''' Use Websearch tool to retrieve information related to topic or subtopic from tavily search.
    This result returns list of dictionary containing websearch results.'''
    search_results=websearch.invoke(topic)
    return search_results

#MIDDLEWARE
middleware_model=ChatGroq(
    model='gemma2-9b-it',
    api_key=SecretStr(groq_api) if groq_api else None,
    temperature=0.1
)

summarization_limit=SummarizationMiddleware(
                model=middleware_model,
                tigger=('tokens',6000),
                keep=('messages',3)
            )

wiki_tool_limit=ToolCallLimitMiddleware(tool_name='wikipedia_retriever_tool',thread_limit=5, run_limit=3)
websearch_tool_limit=ToolCallLimitMiddleware(tool_name='websearch_tool',thread_limit=5, run_limit=3)
# RESEARCH AGENT

def research_agent(state:graphState)-> graphState:
    topic=state.get('topic')

    prompt=SystemMessage(
    content='''Think in step by step method about the information required regrading a praticular topic to generate a summary and research report on it.
    Instructions:-
    1. Tools that you can use: wikipedia_retriever_tool and websearch_tool.(Both the the tools accepts string query)
    2. Keep filtering the data or information you recieve if they are not relevent.
`   3. Information block you retrieve will act as base data for a research paper so it should be detailed.
        It should contain all the sections of a research paper data such as:
        1. abstract
        2. introduction
        3. methodology
        4. results 
        5. discussion
        6. conclusion
        7. references  
        Everything from the above that is relevant to the topic. 
    Note:- 
    1. RETURN A VERY BREIF AND DETAILED INFORMATION ON THE TOPIC.
    2. THE RETURNED INFORMATION SHOULD BETWEEN 5000 TO 10000 WORDS.(WORDS NOT TOKENs)
    3. IN THE END EVEN RETURN THE WORD COUNT OF THE RETURNED TEXT.
    4. NO SPACES BETWEEN PARAGRAPHS
    '''
    )

    tools=[wikipedia_retriever_tool, websearch_tool]

    agent=create_agent(
        model=research_model,
        tools=tools,
        system_prompt=prompt,
        middleware=[
           summarization_limit, wiki_tool_limit, websearch_tool_limit #type:ignore
        ]
    )
    if topic:
        results=agent.invoke({'messages':[HumanMessage(content=topic)]})
        return {'researched_info': results['messages'][-1].content}
    else:
        return {'researched_info':'DATA ERROR'}
    
#SUMMARIZATION AGENT
    
def summarization_agent(state:graphState)->graphState:
    topic= state.get('topic')
    researched_info=state.get('researched_info')
    template='''
    Act as an experet summarizer.
    Instructions:
    1. Generate a consise summary based on the Topic and Researched information.
    2. The summary should be between 5-12 lines.
    3. The summary should contain all important facts and keywords related to the topic.
    4. Provide important insights.

    Topic:{topic}
    Researched information:{researched_info}
    '''
    if topic and researched_info:
        prompt=PromptTemplate(template=template, input_variables=['topic','researched_info'])
        final_prompt=prompt.format_prompt(topic=topic,researched_info=researched_info)
        result=summery_model.invoke(final_prompt)
        return {'summary':str(result.content)}
    else:
        return {'summary':'DATA ERROR'}

#REPORT AGENT
def report_agent(state: graphState)->graphState:
    topic=state.get('topic')
    researched_info=state.get('researched_info')
    summary=state.get('summary')
    template1='''
    YOU ARE AN EXPERT RESEACHER REPORT GENERATOR:
    INSTRUCTIONS:
    1. USING THE FOLLOWING FORMAT TO GENERATE PART OF THE REPORT ON THE TOPIC:{topic}
        1.Abstract (once)
        2.Introduction
        3.Background / Literature Review
        4.Methodology (if required)
    2. THE GENERATED REPORT SHOULD BE PROPERLY FORMATTED.
    3. USE THE RESEARCHED INFORMATION AND SUMMARY PROVIDED AS THE BASE OF THE REPORT:
        SUMMARY: {summary}

        RESEARCHED INFORMATION: {researched_info}

    CONSTRAINTS:
        - FACTS CANNOT BE REPEATED AGAIN
        - USE FORMAL LANGUAGE 
        '''
    template2='''
    YOU ARE AN EXPERT RESEACHER REPORT GENERATOR:
    INSTRUCTIONS:
    1. USING THE FOLLOWING FORMAT TO GENERATE PART OF THE REPORT ON THE TOPIC:{topic}
        1.Results (actual data)
        2.Discussion (interpretation)
        3.Conclusion
        4.References
    2. THE GENERATED REPORT SHOULD BE PROPERLY FORMATTED.
    3. USE THE RESEARCHED INFORMATION AND SUMMARY PROVIDED AS THE BASE OF THE REPORT:
        SUMMARY: {summary}

        RESEARCHED INFORMATION: {researched_info}

         CONSTRAINTS:
        - FACTS CANNOT BE REPEATED AGAIN
        - USE FORMAL LANGUAGE 
        '''

    if researched_info and summary and topic:
        prompt1=PromptTemplate(template=template1, input_variables=['topic','summary','researched_info'])
        prompt2=PromptTemplate(template=template2, input_variables=['topic','summary','researched_info'])
        final_prompt1=prompt1.format_prompt(topic=topic, summary=summary, researched_info=researched_info,)
        final_prompt2=prompt2.format_prompt(topic=topic, summary=summary, researched_info=researched_info,)
        result1=report_model.invoke(final_prompt1)
        result2=report_model.invoke(final_prompt2)
        result=str(result1.content)+" "+str(result2.content)
        return {'report':result}
    else:
        return {'report':'DATA ERROR'}






# Practice query:
# state:graphState={
#     "topic":"Impact of Screen Time on Cognitive Development in Children",
#     'facts':' ',
#     'information':' ',
#     'researched_info':' ',
#     'summary':' ',
#     'report':' ',
#     'result':' '
# }
# results=facts_retrival_agent(state)
# state.update(results)
# results1=information_retrival_agent(state)
# state.update(results1)
# result2=summarization_agent(state)
# state.update(result2)
# result3=report_agent(state)
# state.update(result3)
# print(state)



