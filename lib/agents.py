from typing import TypedDict, List, Optional, Union
import json
import uuid # Import uuid to generate unique session IDs

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory

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
        return {"messages": messages, "session_id": state.get("session_id")}

    def _llm_step(self, state: AgentState) -> AgentState:
        llm = LLM(
            model=self.model_name, temperature=self.temperature, tools=self.tools,
            base_url=self.base_url, api_key=self.api_key
        )
        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None
        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens
        ai_message = AIMessage(content=response.content, tool_calls=tool_calls)
        return {
            "messages": state["messages"] + [ai_message], "current_tool_calls": tool_calls,
            "session_id": state.get("session_id"), "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        tool_calls = state.get("current_tool_calls", [])
        tool_messages = []
        for call in tool_calls:
            function_name = call.function.name
            try:
                function_args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                function_args = {} # Handle cases where arguments are not valid JSON
            tool_call_id = call.id
            tool = next((t for t in self.tools if t.name == function_name), None)
            if tool:
                result = tool(**function_args)
                tool_message = ToolMessage(
                    content=json.dumps(result, default=str), # Use default=str for broader serialization
                    tool_call_id=tool_call_id, name=function_name,
                )
                tool_messages.append(tool_message)
        return {
            "messages": state["messages"] + tool_messages, "current_tool_calls": None,
            "session_id": state.get("session_id")
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        machine = StateMachine[AgentState](AgentState)
        entry, message_prep, llm_processor, tool_executor, termination = (
            EntryPoint[AgentState](), Step[AgentState]("message_prep", self._prepare_messages_step),
            Step[AgentState]("llm_processor", self._llm_step), Step[AgentState]("tool_executor", self._tool_step),
            Termination[AgentState]()
        )
        machine.add_steps([entry, message_prep, llm_processor, tool_executor, termination])
        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)
        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            return tool_executor if state.get("current_tool_calls") else termination
        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(tool_executor, llm_processor)
        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a query.
        - If session_id is provided, it maintains conversation state.
        - If session_id is None, it runs as a clean, single-shot query.
        """
        # *** THIS IS THE FINAL FIX ***
        # If no session_id is given, create a unique, temporary one for this run only.
        # This ensures stateless (non-conversational) queries are always isolated.
        is_new_session = session_id is None
        session_id = session_id or f"temp_session_{uuid.uuid4()}"
        
        self.memory.create_session(session_id)
        
        previous_messages = []
        # Only load previous messages if it's an existing, ongoing conversation.
        if not is_new_session:
            last_run: Run = self.memory.get_last_object(session_id)
            if last_run:
                last_state = last_run.get_final_state()
                if last_state and last_state.get("messages"):
                    previous_messages = last_state["messages"]
        
        initial_state: AgentState = {
            "user_query": query, "instructions": self.instructions, "messages": previous_messages,
            "current_tool_calls": None, "session_id": session_id, "total_tokens": 0
        }
        
        run_object = self.workflow.run(initial_state)
        # Only add to memory if it's a persistent session, not a temporary one.
        if not is_new_session:
            self.memory.add(run_object, session_id)
            
        return run_object
