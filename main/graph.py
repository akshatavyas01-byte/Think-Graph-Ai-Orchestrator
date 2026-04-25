from langgraph.graph import START, StateGraph
from state import graphState
from node import facts_retrival_agent, information_retrival_agent,summarization_agent,report_agent,feedback_agent, router_node

graph=StateGraph(graphState)

graph.add_node('fact_node',facts_retrival_agent)
graph.add_node('information_node',information_retrival_agent)
graph.add_node('summary_node',summarization_agent)
graph.add_node('report_node',report_agent)
graph.add_node('feedback_node',feedback_agent)
graph.add_node('router_node',router_node)


graph.add_edge(START, 'fact_node')
graph.add_edge('fact_node','information_node')
graph.add_edge('information_node','summary_node')
graph.add_edge('summary_node','report_node')
graph.add_edge('report_node','feedback_node')
graph.add_edge('feedback_node','router_node')


AI_researcher=graph.compile()


#Practice Query

# state:graphState={
#   "topic":"Impact of Screen Time on Cognitive Development in Children"
#   }

# result=AI_researcher.invoke(state)
# summary=result.get('summary')
# report=result.get('report')
# feedback=result.get('feedback')
# print("SUMMARY:\n", summary or "Missing summary")
# print("\nREPORT:\n", report or "Missing report")
# print("\nFEEDBACK:\n", feedback or "Missing feedback")
