from typing import Annotated, Optional, Dict, Any
from typing_extensions import TypedDict
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class StateSchema(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]
    route: Optional[str]

    user_data: Optional[Dict[str, Any]]
    debug: Optional[Any]
    confirmation: bool
    
    



    