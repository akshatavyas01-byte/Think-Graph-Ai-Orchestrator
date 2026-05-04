from typing import TypedDict, Literal

class graphState(TypedDict, total=False):
    topic:str
    facts:str
    information:str
    researched_info: str
    summary:str
    report:str
    feedback:str
    router_result:Literal['Pass','Fail']