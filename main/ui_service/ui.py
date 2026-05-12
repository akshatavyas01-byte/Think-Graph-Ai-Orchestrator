import streamlit as st
import httpx 

if "result" not in st.session_state:
    st.session_state.result=None

if "error" not in st.session_state:
    st.session_state.error = None


def next():
    st.session_state.clear()
    st.session_state.topic_clear=" "
    st.rerun()

st.title("Think Graph")

topic=st.text_area("Enter the Topic", key="topic_clear")

placeholder=st.empty()

task=st.radio(
    "Select the task:",
    ["Summary","Information","Full Report"]
    )


def request(topic:str, task:str, resubmit=False):
    try:
        with st.spinner(text="Loading",show_time=False):
            if task=="Summary":
                url="https://think-graph-ai-orchestrator.onrender.com/reseacher/summary_generator"
                
            elif task=="Information":
                url="https://think-graph-ai-orchestrator.onrender.com/reseacher/information_retrival"
               
            elif task=="Full Report":
                url="https://think-graph-ai-orchestrator.onrender.com/reseacher/report_generator"
                
            else:
                placeholder.write("SELECTION ERROR")
                return 
            if task=="Information":
                response=httpx.get(
                        url=url,
                        params={"topic":topic},
                        timeout=600.0
                    )
            else :
                response=httpx.get(
                        url=url,
                        params={"topic":topic, "resubmit":resubmit},
                        timeout=600.0
                    )
            response.raise_for_status()
            data = response.json()
            st.session_state.result=data
            
    except Exception as e:
       st.session_state.error=str(e)


if st.button("Submit"):
    request(topic,task)

def display_func(title, content):
    if isinstance(content,list):
        text="\n\n".join(f'- {item}' for item in content)
    else:
        text=str(content)
    placeholder.markdown(f"#### {title}\n{text}")

def display_report(report_content):
    text=""
    counter=1
    if isinstance(report_content, list):
        for values in (report_content):
            summary=values.get("Summary")
            report=values.get("Report")
            feedback=values.get("Feedback")
            text+=f"\n #### Report {counter}\n\n ##### Summary \n\n {summary}\n\n #### Report \n\n {report}\n\n ##### Feedback \n\n {feedback}\n\n"
            counter+=1
        placeholder.markdown(text)
    else:
        summary=report_content.get("Summary")
        report=report_content.get("Report")
        feedback=report_content.get("Feedback")
        text+=f"\n #### Report \n\n ##### Summary \n\n {summary}\n\n #### Report \n\n {report}\n\n ##### Feedback \n\n {feedback}\n\n"
        placeholder.markdown(text)


if st.session_state.result:
    data=st.session_state.get("result")
    print(type(data))
    if isinstance(data,dict):
        print(data)
        if all(k in data for k in ["Summary","Report","Feedback"]):
            display_report(data)
        elif "Summary" in data:
            display_func("Summary",data.get("Summary"))

        elif "Information" in data:
            display_func("Information",data.get('Information'))

    elif isinstance(data, list):
        display_report(data)

if st.button("NEXT"):
        next()
 
if st.button("RESUBMIT"):
    request(topic, task, True)
    st.rerun()


if st.session_state.error:
    st.error(st.session_state.get("error"))



