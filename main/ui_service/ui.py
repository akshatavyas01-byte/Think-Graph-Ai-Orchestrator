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


def request(topic:str, task:str):
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

            response=httpx.get(
                    url=url,
                    params={"topic":topic},
                    timeout=600.0
                )
            response.raise_for_status()
            data = response.json()
            print(data)
            st.session_state.result=data
            
    except Exception as e:
       st.session_state.error=str(e)


if st.button("Submit"):
    request(topic,task)


if st.session_state.result:
    data=st.session_state.get("result")
    print(data)
    if all(k in data for k in ["Summary","Report","Feedback"]):
        placeholder.markdown(f"#### Summary: \n{data.get('Summary')}\n #### Report:\n{data.get('Report')}\n #### Feedback: \n{data.get('Feedback')}")
    elif "Summary" in data:
        print(data.get('Summary'))
        placeholder.markdown(f"#### Summary: \n {data.get('Summary')}")
    elif "Information" in data:
        placeholder.markdown(f"#### Information:\n {data.get('Information')}")
   
    
if st.button("NEXT"):
        next()

if st.session_state.error:
    st.error(st.session_state.get("error"))



