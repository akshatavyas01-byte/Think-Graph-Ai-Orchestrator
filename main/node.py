from state import graphState
from langgraph.types import Command
from langgraph.graph import END
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
import time 
# from langchain_core.vectorstores import FAISS
from langchain_groq import ChatGroq
from pydantic import SecretStr
from dotenv import load_dotenv
import os

load_dotenv()

#LLM MODEL BLOCK
hf_api=os.getenv("hf_api")

groq_api=os.getenv("groq_api")


facts_retrival_model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(groq_api) if groq_api else None,
    temperature=0.1 
)

information_retrival_model=ChatGroq(
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

feedback_model=ChatHuggingFace(llm=llm)

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

# summarization_limit=SummarizationMiddleware(
#                 model=middleware_model,
#                 tigger=('tokens',6000),
#                 keep=('messages',3)
#             )

wiki_tool_limit=ToolCallLimitMiddleware(tool_name='wikipedia_retriever_tool',thread_limit=5, run_limit=5)
websearch_tool_limit=ToolCallLimitMiddleware(tool_name='websearch_tool',thread_limit=5, run_limit=5)
# FACTS RETRIVAL AGENT

def facts_retrival_agent(state:graphState)-> graphState:
    topic=state.get('topic')
    router_result=state.get('router_result')
    previous_facts=state.get('facts')
    
    prompt=SystemMessage(
    content='''
    You are a research agent.
    Your job is to:
    1. Use BOTH tools: wikipedia_retriever_tool and websearch_tool
    2. Retrieve relevant, high-quality information
    3. Extract factual, detailed content from sources

    STRICT RULES:
    - You MUST use tools before answering
    - Do NOT rely on prior knowledge
    - Only use retrieved information
    - Include references for every major point

    OUTPUT FORMAT:
    - Bullet points or structured paragraphs
    - Include:
    • Key facts
    • Explanations
    • Source (link + title)

    CONSTRAINTS:
    - Keep response between 800-1500 words
    - Do NOT include word count
    - Focus on depth, not length
    '''
    )
    prompt_template=prompt
    prompt2=SystemMessage( 
        content=f'''
        You are a research agent.
        Your job is to:
            1. Use BOTH tools: wikipedia_retriever_tool and websearch_tool
            2. Retrieve relevant, high-quality information
            3. Extract factual, detailed content from sources
            4. It should Not include the previously fectched facts
                5. The Facts retrieve now should be relevent and unique

            STRICT RULES:
            - You MUST use tools before answering
            - Do NOT rely on prior knowledge
            - DO NOT include the prevoiusly fectched facts
            - Each fact must be unique
            - Only use retrieved information
            - Include references for every major point

            OUTPUT FORMAT:
            - Bullet points or structured paragraphs
            - Include:
            • Key facts
            • Explanations
            • Source (link + title)

            CONSTRAINTS:
            - Keep response between 800-1500 words
            - Focus on depth, not length
            PREVIOUS FACTS:{previous_facts}
            '''
            )
    
    tools=[wikipedia_retriever_tool, websearch_tool]

    if topic and previous_facts and router_result:
        prompt_template=prompt2

    agent=create_agent(
        model=facts_retrival_model,  
        tools=tools,
        system_prompt=prompt_template,
        middleware=[
            wiki_tool_limit, websearch_tool_limit #type:ignore
            # summarization_limit,
        ]
    )
    if topic:
        results=agent.invoke({'messages':[HumanMessage(content=topic)]})
        return {'facts': results['messages'][-1].content}
    else:
        return {'facts':'DATA ERROR'}
    
# INFORMATION RETRIVAL AGENT 

def information_retrival_agent(state:graphState)->graphState:
    facts=state.get('facts')
    template='''
    You are a research agent.
    Your job is to:
    1. Retrieve relevant, high-quality information.
    2. Research and generate the detailed information about the facts.
    3. Information regrading the facts and meaningful insights regrading them.

    Format:
    - Fact
    - Detailed information
    - Meaningful insights 

    Facts:
    {facts}
    '''
    if facts:
        prompt=PromptTemplate(template=template,input_variables=['facts'])
        final_prompt=prompt.format(facts=facts)
        result=information_retrival_model.invoke(final_prompt)
        researched_info=facts+" "+str(result.content)
        return {'information':str(result.content),'researched_info':researched_info}
    else:
        return {'information':'DATA ERROR'}

    
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
    1. WRITE ONLY THE FOLLOWING SECTIONS FOR THE TOPIC:{topic}
        1.Abstract (once)
        2.Introduction
        3.Background / Literature Review
        4.Methodology (if required)
    2. DO NOT INCLUDE:
        - Results (actual data)
        - Discussion (interpretation)
        - Conclusion
        - References
    3. THE GENERATED REPORT SHOULD BE PROPERLY FORMATTED.
    3. USE THE RESEARCHED INFORMATION AND SUMMARY PROVIDED AS THE BASE OF THE REPORT:
        SUMMARY:
        {summary}

        RESEARCHED INFORMATION: 
        {researched_info}

    CONSTRAINTS:
        - DO NOT REPEAT FACTS UNNECESSARILY
        - USE FORMAL ACADEMIC LANGUAGE 
        '''
    template2='''
    YOU ARE AN EXPERT RESEACHER REPORT GENERATOR:
    INSTRUCTIONS:
    1. WRITE ONLY THE FOLLOWING SECTIONS FOR THE TOPIC:{topic}
        1.Results (actual data)
        2.Discussion (interpretation)
        3.Conclusion
        4.References
    2. DO NOT INCLUDE:
        - Abstract (once)
        - Introduction
        - Background / Literature Review
        - Methodology (if required)
    2. THE GENERATED REPORT SHOULD BE PROPERLY FORMATTED.
    3. SEAMLESSLY MERGE WITH THE  PERVIOUS SECTION AND USE THE RESEARCHED INFORMATION OF REPORT AS THE BASE OF THE REPORT:
        PREVIOUS SECTION:
         {section1}

        RESEARCHED INFORMATION: {researched_info}

    CONSTRAINTS:
        - DO NOT repeat earlier content
        - FACTS CANNOT BE REPEATED AGAIN
        - USE FORMAL LANGUAGE 
        - NOT RESTART OR DUPLICATE EARLIER SECTIONS
        - AVOID REPEATING FACTS ALREADY USED
    '''

    if researched_info and summary and topic:
        prompt1=PromptTemplate(template=template1, input_variables=['topic','summary','researched_info'])
        prompt2=PromptTemplate(template=template2, input_variables=['topic','section1','researched_info'])
        final_prompt1=prompt1.format_prompt(topic=topic, summary=summary, researched_info=researched_info,)
        result1=report_model.invoke(final_prompt1)
        section1=str(result1.content)
        final_prompt2=prompt2.format_prompt(topic=topic, section1=section1, researched_info=researched_info,)
        time.sleep(5)
        result2=report_model.invoke(final_prompt2)
        result=str(result1.content)+" "+str(result2.content)
        return {'report':result}
    else:
        return {'report':'DATA ERROR'}


#FEEDBACK MODEL
def feedback_agent(state:graphState)->graphState:
    report=state.get('report')
    template='''
    YOU ARE AN EXPERT REPORT REVIEWER
    INSTRUCTIONS:
        - REVIEW THE REPORT 
        - RATE THE REPORT OUT OF 10

    OUTPUT FROMAT:
        - ONLY THE NUMBER 
        - NO EXPLAINATION OR REASONS

    EXAMPLE OUTPUT:
        5
    
    REPORT:
    {report}
        
    '''

    if report:
        prompt=PromptTemplate(template=template, input_variables=['report'])
        final_prompt=prompt.format(report=report)
        result=feedback_model.invoke(final_prompt)
        return {'feedback':str(result.content)}
    else:
        return {'feedback':'DATA ERROR'}

#ROUTER NODE
def router_node(state:graphState):
    feedback=state.get('feedback')
    if feedback:
        value=int(feedback)
        if value>7:
            return Command(
                update={'router_result':'PASS'},
                goto=END
            )
        else:
            return Command(
                update={'router_result':'Fail'},
                goto='fact_node'
            )
    else:
        return Command(
                update={'router_result':'Fail'},
                goto='fact_node'
            )

# Practice query:
# state:graphState={
#     "topic":"Impact of Screen Time on Cognitive Development in Children",
# }
# results=facts_retrival_agent(state)
# state.update(results)
# results1=information_retrival_agent(state)
# state.update(results1)
# result2=summarization_agent(state)
# state.update(result2)
# result3=report_agent(state)
# state.update(result3)
# result4=feedback_agent(state)
# state.update(result4)
# result5=router_node(state)
# state.update(result5)
# print(state)



