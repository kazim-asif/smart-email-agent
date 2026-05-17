# smart-email-agent
An intelligent email automation agent built with LangGraph, featuring structured LLM classification, dynamic command-based routing, and Human-in-the-Loop (HITL) review for high-urgency customer support tickets. Utilizes memory persistence, structured outputs via ChatOpenAI, and state machine interrupts for manual human approval.

## How It Works (The Workflow)

Ingestion & Classification: Reads the incoming email and extracts its topic, intent (e.g., bug, billing), and urgency level using OpenAI structured outputs.

Draft Generation: Automatically writes a targeted response tailored to the extracted intent and urgency guidelines.

Risk-Based Routing: Uses LangGraph Command objects to evaluate the ticket. Low-risk emails go straight to the queue, while high-urgency or complex issues trigger an immediate interrupt().

State Checkpointing: Pauses execution and securely saves the exact graph state using InMemorySaver memory persistence while awaiting human input.

Human-in-the-Loop Review: Resumes the graph once a human operator approves, edits, or rejects the draft via Command(resume=...) to finalize or kill the process.


## Learning Outcomes

**State Machine Architecture**: Shifted from linear chains to a flexible graph network using LangGraph StateGraph.

**State Persistence & Checkpoints**: Implemented InMemorySaver to act as a flight recorder, enabling the graph to safely pause and resume across human interventions without data loss.

**Dynamic Node Routing**: Utilized inline Command objects to handle both state updates and conditional routing (goto) natively inside the execution nodes.

**Human-in-the-Loop (HITL)**: A non-blocking asynchronous patterns using interrupt() to await external human feedback.

**Structured Data Extraction**: Applied with_structured_output to guarantee the LLM adheres strictly to a predefined schema for reliable routing logic.
