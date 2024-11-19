from langchain import hub
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from component.agent import PlanningAgent

def client_setup():
    client = Client()

    examples = [("What are the schedules on December 1st, 2024?", "You have a project kickoff meeting and quarterly results review."),
("Do I have any schedules on December 2nd, 2024?", "Yes, you have a team building exercise and client follow-up call."),
("What is planned for December 3rd, 2024?", "You have an update project roadmap and preparing marketing strategy."),
("Are there any meetings on December 4th, 2024?", "Yes, there is a budget planning session and technical workshop."),
("What activities are scheduled for December 5th, 2024?", "You have a sales team meeting and product launch review."),
("Is there anything planned for December 6th, 2024?", "Yes, there is a monthly finance audit and organizing office event."),
("What is the schedule for December 7th, 2024?", "You have a weekly team sync and customer feedback analysis."),
("Do I have any meetings on December 8th, 2024?", "Yes, there is social media planning and new hire orientation."),
("What is happening on December 9th, 2024?", "You have system maintenance and reviewing compliance policies."),
("Are there any events on December 10th, 2024?", "Yes, there is preparing annual report and internal training session."),
("What is planned for December 11th, 2024?", "You have evaluating software tools and holiday party planning."),
("Do I have any meetings on December 12th, 2024?", "Yes, there is a supplier meeting and inventory assessment."),
("What is scheduled for December 13th, 2024?", "You have a board of directors meeting and industry conference."),
("Are there any activities on December 14th, 2024?", "Yes, there is researching competitor strategies and project milestone review."),
("What is happening on December 15th, 2024?", "You have updating employee handbook and IT security audit."),
("Do I have any meetings on December 16th, 2024?", "Yes, there is preparing for product demo and conducting market survey."),
("What is planned for December 17th, 2024?", "You have a team retrospective and finalizing partnership agreement."),
("Are there any events on December 18th, 2024?", "Yes, there is planning community outreach and reviewing team performance."),
("What is scheduled for December 19th, 2024?", "You have an annual strategy meeting and updating corporate website."),
("Do I have any meetings on December 20th, 2024?", "Yes, there is design sprint planning and office renovation discussion."),
("What is happening on December 21st, 2024?", "You have preparing for trade show and product feedback session."),
("Are there any activities on December 22nd, 2024?", "Yes, there is holiday season preparation and legal compliance review."),
("What is planned for December 23rd, 2024?", "You have finalizing year-end bonuses and planning for office relocation."),
("Do I have any meetings on December 24th, 2024?", "No, it is Christmas Eve and the office is closed."),
("What is happening on December 25th, 2024?", "It is Christmas Day and the office is closed."),
("Are there any events on December 26th, 2024?", "Yes, there is a post-holiday debrief."),
("What is scheduled for December 27th, 2024?", "You have a year-end financial review."),
("Do I have any meetings on December 28th, 2024?", "No, you are free on December 28th, 2024."),
("What is planned for December 29th, 2024?", "You are free on December 29th, 2024."),
("Are there any events on December 30th, 2024?", "No, you are free on December 30th, 2024."),
("What is happening on December 31st, 2024?", "You are free on December 31st, 2024."),
("How many events are scheduled in December 2024?", "There are 50 events scheduled in December 2024."),
("Are there any meetings on weekends in December 2024?", "No, there are no meetings scheduled on weekends in December 2024."),
("What is the most frequent type of event in December 2024?", "The most frequent type of event is meetings."),
("Are there any consecutive days with multiple events in December 2024?", "Yes, several days have multiple events, such as December 1st, 2nd, 3rd, etc."),
("What is the longest gap between events in December 2024?", "The longest gap is between December 25th and December 26th, 2024."),
("Are there any events in the last week of December 2024?", "Yes, there are events from December 23rd to December 27th, 2024."),
("What is the earliest event in December 2024?", "The earliest event is on December 1st, 2024."),
("Are there any events in the first week of December 2024?", "Yes, there are events from December 1st to December 7th, 2024."),
("What is the latest event in December 2024?", "The latest event is on December 27th, 2024."),
("Do I have any meetings in the second week of December 2024?", "Yes, there are meetings from December 8th to December 14th, 2024."),
("Are there any events in the third week of December 2024?", "Yes, there are events from December 15th to December 21st, 2024."),
("What is the most common day of the week for events in December 2024?", "The most common day of the week for events is Tuesday."),
("Are there any events in the fourth week of December 2024?", "Yes, there are events from December 22nd to December 27th, 2024."),
("What is the total number of meetings in December 2024?", "There are 5 meetings in December 2024."),
("Are there any events on December 10th, 2024?", "Yes, there is preparing annual report and internal training session."),
("What is the total number of events in December 2024?", "There are 50 events in December 2024."),
("Are there any meetings on December 15th, 2024?", "Yes, there is updating employee handbook and IT security audit."),
("What is the total number of events in the first half of December 2024?", "There are 30 events in the first half of December 2024."),
("Are there any meetings on December 20th, 2024?", "Yes, there is design sprint planning and office renovation discussion.")]

    dataset_name = "Schedule Info Query Agent"
    dataset = client.create_dataset(dataset_name=dataset_name, description="QA pairs of Schedule Info")
    inputs, outputs = zip(*[({"input": text}, {"output": label}) for text, label in examples])
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

def answer_evaluator(run, example) -> dict:
    grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")

    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs["response"]

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    answer_grader = grade_prompt_answer_accuracy | llm

    score = answer_grader.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "similarity_score", "score": score}

def predict_sql_agent_answer(example: dict):
    planning_agent = PlanningAgent()
    planning_agent_response = planning_agent.full_chain.invoke({"input": example["input"]})["output"]
    return {"response": planning_agent_response}

def main():
    client_setup()

    experiment_results = evaluate(
        predict_sql_agent_answer,
        data="Schedule Info Query Agent",
        evaluators=[answer_evaluator],
        experiment_prefix="sql-agent-gpt4o" + "-similarity_score",
        num_repetitions=3,
        metadata={"version": "v1.1"},
    )

    print(experiment_results)

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    main()