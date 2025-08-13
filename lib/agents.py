from typing import TypedDict, List, Optional, Union, TypeVar
import json

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory

# Define the state schema
class AgentState(TypedDict):
    user_query: str
    instructions: str
    messages: List[dict]
    current_tool_calls: Optional[List[ToolCall]]
    total_tokens: int
    
class Agent:
    def __init__(self, 
                 model_name: str,
                 instructions: str, 
                 tools: List[Tool] = None,
                 temperature: float = 0.7,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        
        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]
            
        messages.append(UserMessage(content=state["user_query"]))
        
        return {
            "messages": messages,
            "session_id": state.get("session_id")
        }

    def _llm_step(self, state: AgentState) -> AgentState:
        # Correctly initializes the LLM with all necessary parameters
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            tools=self.tools,
            base_url=self.base_url,
            api_key=self.api_key
        )

        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        ai_message = AIMessage(
            content=response.content, 
            tool_calls=tool_calls,
        )

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state.get("session_id"),
            "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        tool_calls = state["current_tool_calls"] or []
        tool_messages = []
        
        for call in tool_calls:
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id
            tool = next((t for t in self.tools if t.name == function_name), None)
            
            if tool:
                result = tool(**function_args)
                tool_message = ToolMessage(
                    content=json.dumps(result, default=lambda o: o.__dict__), # Handle complex objects
                    tool_call_id=tool_call_id, 
                    name=function_name, 
                )
                tool_messages.append(tool_message)
        
        return {
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state.get("session_id")
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        machine = StateMachine[AgentState](AgentState)
        
        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step)
        termination = Termination[AgentState]()
        
        machine.add_steps([entry, message_prep, llm_processor, tool_executor, termination])
        
        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)
        
        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            if state.get("current_tool_calls"):
                return tool_executor
            return termination
        
        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(tool_executor, llm_processor)
        
        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        session_id = session_id or "default"
        self.memory.create_session(session_id)
        
        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state.get("messages", [])

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
            "total_tokens": 0
        }

        run_object = self.workflow.run(initial_state)
        self.memory.add(run_object, session_id)
        
        return run_object
