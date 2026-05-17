import uuid
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI


load_dotenv()

class EmailClassification(TypedDict):
    topic: str
    summary: str
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]

class AgentState(TypedDict):
    sender_email: str
    email_subject: str
    email_body: str

    classification: EmailClassification

    draft_response: str

builder = StateGraph(AgentState)

llm = ChatOpenAI(model="gpt-5-mini")

def read_email(state: AgentState) -> AgentState:
    return state

def classify_email(state: AgentState) -> AgentState:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    # Create structured LLM that returns EmailClassification dict
    structured_llm = llm.with_structured_output(EmailClassification)

    classification_prompt = f"""
        Analyze this customer email and classify it:

        Subject: {state['email_subject']}
        Email: {state['email_body']}
        From: {state['sender_email']}

        Provide classification, including intent, urgency, topic, and summary
        """

    # Get structured response directly as a dict
    classification = structured_llm.invoke(classification_prompt)

    # Store classification as a single dict in state
    return {"classification": classification}

def write_response(state: AgentState) -> Command[Literal["human_review", "send_reply"]]:
    classification = state.get('classification', {})
    intent = classification.get('intent', 'unknown')
    urgency = classification.get('urgency', 'medium')

    # Build the prompt with formatted context
    draft_prompt = f"""
        Draft a response to this customer email:
        {state['email_subject']}
        {state['email_body']}
        
        Email intent: {intent}
        Urgency level: {urgency}

        Guidelines:
        - Be professional and helpful
        - Address their specific concern
        - Be brief
        """

    response = llm.invoke(draft_prompt)

    needs_review = (
            urgency in ('high', 'critical') or
            intent == 'complex'
    )

    # Route to the appropriate next node
    if needs_review:
        goto = "human_review"
        print("Needs approval")
    else:
        goto = "send_reply"

    return Command(
        update = {"draft_response": response.content},
        goto = goto
    )

def human_review(state: AgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    classification = state.get('classification', {})

    # Interrupt() must come first - any code before it will re-run on resume
    human_decision = interrupt({
        "sender_email": state['sender_email'],
        "email_subject": state['email_subject'],
        "original_email": state['email_body'],
        "draft_response": state.get('draft_response', ""),
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "Please review and approve/edit this response"
    })

    # Now process the human's decision
    if human_decision.get("approved"):
        return Command(
            update = {"draft_response": human_decision.get("edited_response", state['draft_response'])},
            goto = "send_reply"
        )
    else:
        # Rejection means human will handle directly
        return Command(update = {}, goto = END)

def send_reply(state: AgentState) -> AgentState:
    """Send the email response"""
    # Integrate with a email service
    print(f"Sending reply: {state['draft_response'][:60]}...")
    return {}

# Add nodes
builder.add_node("read_email", read_email)
builder.add_node("classify_email", classify_email) # Named 'classify_email' here
builder.add_node("write_response", write_response)
builder.add_node("human_review", human_review)
builder.add_node("send_reply", send_reply)

# Add edges
builder.add_edge(START, "read_email")
builder.add_edge("read_email", "classify_email")   # Changed from classify_intent
builder.add_edge("classify_email", "write_response") # Changed from classify_intent
builder.add_edge("send_reply", END)

from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
app = builder.compile(checkpointer = memory)

# Test with urgent billing issue
initial_state = {
    "email_subject": "Charged twice for subscription",
    "email_body": "I was charged twice for my subscription! This is urgent!",
    "sender_email": "customer@example.com",
}
needs_approval = []

thread_id = uuid.uuid4()
config =  {"configurable": {"thread_id": thread_id}}

# Initial invocation (This will run up to the interrupt because it's high urgency)
result = app.invoke(initial_state, config)

# Check if the graph is paused at an interrupt
if "__interrupt__" in result.keys():
    print("\n--- Graph Paused for Human Review ---")

    human_feedback = {
        "approved": True,
        "edited_response": "Dear customer, we apologize for the double charge. We have initiated a refund. Thank you!"
    }

    print(f"Human Decision: Approved with custom response.")

    # 3. RESUME THE GRAPH by passing the decision into the command
    final_result = app.invoke(
        Command(resume=human_feedback),  # This feeds directly into the 'human_decision' variable
        config
    )
    print("\n--- Graph Resumed and Completed ---")