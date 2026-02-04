import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# --- Configuration ---
# Ensure these are set in your environment or .env file
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    print("❌ Error: Missing Azure OpenAI Environment Variables.")
    print("Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    sys.exit(1)

# Initialize Azure LLM
llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_version=AZURE_API_VERSION,
    temperature=0.3
)

# --- State Management ---
@dataclass
class InterviewState:
    candidate_name: str
    job_description: str
    analyzed_requirements: Dict = field(default_factory=dict)
    generated_questions: List[str] = field(default_factory=list)
    evaluations: List[Dict] = field(default_factory=list)
    final_recommendation: Optional[str] = None

# --- Agent 1: Job Description Analyzer ---
class JobAnalysis(BaseModel):
    role_title: str = Field(description="The title of the role")
    technical_skills: List[str] = Field(description="List of required technical skills")
    soft_skills: List[str] = Field(description="List of required soft skills")
    seniority_level: str = Field(description="Junior, Mid, Senior, or Lead")

jd_parser = JsonOutputParser(pydantic_object=JobAnalysis)
jd_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Expert Technical Recruiter. Analyze the job description and extract key requirements.\n{format_instructions}"),
    ("human", "Job Description:\n{jd}")
])
jd_agent = jd_prompt | llm | jd_parser

def analyze_jd(state: InterviewState):
    print(f"🕵️  Agent: Analyzing Job Description...")
    result = jd_agent.invoke({
        "jd": state.job_description,
        "format_instructions": jd_parser.get_format_instructions()
    })
    state.analyzed_requirements = result
    print(f"    -> Role Identified: {result['role_title']}")
    return state

# --- Agent 2: Question Generator ---
q_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Technical Interviewer. Generate ONE tough but fair interview question."),
    ("system", "Role: {role}\nTech Skills: {tech_skills}\nSeniority: {seniority}"),
    ("system", "Avoid repeating these questions: {previous_questions}"),
    ("human", "Generate a question focusing on: {focus_area}")
])
q_agent = q_prompt | llm | StrOutputParser()

def generate_question(state: InterviewState, topic: str):
    response = q_agent.invoke({
        "role": state.analyzed_requirements.get('role_title'),
        "tech_skills": ", ".join(state.analyzed_requirements.get('technical_skills', [])),
        "seniority": state.analyzed_requirements.get('seniority_level'),
        "previous_questions": "\n".join(state.generated_questions),
        "focus_area": topic
    })
    state.generated_questions.append(response)
    return response

# --- Agent 3: Answer Evaluator ---
class EvaluationScore(BaseModel):
    score: int = Field(description="Score from 1-10")
    reasoning: str = Field(description="Brief explanation of the score")

eval_parser = JsonOutputParser(pydantic_object=EvaluationScore)
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Engineer evaluating a candidate answer. Be strict but constructive.\n{format_instructions}"),
    ("human", "Question: {question}\nCandidate Answer: {answer}\nEvaluate relevance to these skills: {skills}")
])
eval_agent = eval_prompt | llm | eval_parser

def evaluate_answer(state: InterviewState, question: str, answer: str):
    result = eval_agent.invoke({
        "question": question,
        "answer": answer,
        "skills": state.analyzed_requirements.get('technical_skills'),
        "format_instructions": eval_parser.get_format_instructions()
    })
    state.evaluations.append({
        "question": question, "answer": answer,
        "score": result['score'], "reasoning": result['reasoning']
    })
    return result

# --- Agent 4: Hiring Manager ---
rec_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Hiring Manager. Make a final hiring decision based on the interview logs."),
    ("human", "Candidate: {name}\nRole: {role}\n\nInterview Performance Summary:\n{evaluations}\n\nProduce a final hiring recommendation (HIRE, NO HIRE, or FOLLOW-UP).")
])
rec_agent = rec_prompt | llm | StrOutputParser()

def generate_final_decision(state: InterviewState):
    print("\n🏆 Agent: Synthesizing final recommendation...")
    eval_summary = ""
    for entry in state.evaluations:
        eval_summary += f"Q: {entry['question']}\nScore: {entry['score']}/10\nNotes: {entry['reasoning']}\n\n"
        
    decision = rec_agent.invoke({
        "name": state.candidate_name,
        "role": state.analyzed_requirements.get('role_title'),
        "evaluations": eval_summary
    })
    state.final_recommendation = decision
    return decision

# --- Orchestrator ---
def run_interview(jd_text: str, candidate_name: str):
    state = InterviewState(candidate_name=candidate_name, job_description=jd_text)
    
    # 1. Analyze
    analyze_jd(state)
    skills = state.analyzed_requirements.get('technical_skills', [])[:2] # Limit to 2 for demo

    print(f"\n--- 🎤 Starting Interview for {candidate_name} ---")
    
    # 2. Loop through topics
    for skill in skills:
        print(f"\n🤖 Agent: Generating question about {skill}...")
        question = generate_question(state, topic=skill)
        print(f"AI Interviewer: {question}")
        
        # Interactive Input
        answer = input("Your Answer: ")
        
        print("⚖️  Agent: Evaluating answer...")
        evaluation = evaluate_answer(state, question, answer)
        print(f"   -> Score: {evaluation['score']}/10")

    # 3. Final Decision
    final_verdict = generate_final_decision(state)
    print(f"\n📝 FINAL DECISION:\n{final_verdict}")

if __name__ == "__main__":
    sample_jd = """
    We are looking for a Senior Python Developer. 
    Must have experience with Django, REST APIs, and Azure Cloud services.
    Knowledge of LangChain and AI Agents is a huge plus.
    """
    run_interview(sample_jd, "Alex Developer")
