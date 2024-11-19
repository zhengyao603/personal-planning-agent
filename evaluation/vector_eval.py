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

    examples = [("What is machine learning?", "A subfield of artificial intelligence that focuses on building systems capable of learning from data."),
("What are the two main types of learning in machine learning?", "Supervised and unsupervised learning."),
("What does supervised learning involve?", "Training a model on a labeled dataset where the output is known."),
("Name two common algorithms used in supervised learning.", "Linear regression and decision trees."),
("What is unsupervised learning?", "It deals with unlabeled data and finds hidden patterns or structures."),
("What are popular techniques in unsupervised learning?", "Clustering and association."),
("Why is data quality important in machine learning?", "Because garbage in results in garbage out."),
("What are feature selection and extraction?", "Vital steps that enhance model performance."),
("Which library is mentioned for implementing ML algorithms?", "Python's scikit-learn library."),
("How is machine learning transforming industries?", "By enabling data-driven decision-making."),
("What is the core idea of machine learning?", "To use algorithms to parse data, learn from it, and make informed decisions."),
("What is cloud computing?", "A paradigm that enables on-demand network access to configurable computing resources."),
("What resources does cloud computing provide?", "Servers, storage, applications, and services."),
("Name the three main service models of cloud computing.", "Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS)."),
("What does IaaS provide?", "Virtualized computing resources over the internet."),
("What is the purpose of PaaS?", "To allow customers to develop, run, and manage applications."),
("What does SaaS deliver?", "Software applications over the web."),
("List some benefits of cloud computing.", "Cost efficiency, scalability, flexibility, and improved collaboration."),
("What are some challenges of cloud computing?", "Security concerns and potential downtime."),
("Name three major cloud service providers.", "Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform."),
("Why is understanding cloud services critical?", "For leveraging digital transformation."),
("What is Golang?", "A statically typed, compiled language designed by Google for backend development."),
("What makes Golang ideal for backend development?", "Its simplicity, efficiency, and robust concurrency support."),
("Describe Go's syntax.", "Concise and easy to understand."),
("What powers Golang's concurrency model?", "Goroutines and channels."),
("Why is performance a strength of Golang?", "Its compiled nature provides performance comparable to C/C++."),
("What tasks does Golang's standard library simplify?", "HTTP handling, JSON parsing, and database interaction."),
("Name two frameworks for Golang.", "Gin and Echo."),
("What are the benefits of using Golang for development?", "Performance, simplicity, and scalability."),
("What is the aim of learning Golang according to the memo?", "To build a small web service to apply the concepts practically."),
("Who designed Golang?", "Google."),
("What is a crucial feature for backend systems in Golang?", "Efficiently handling multiple tasks simultaneously."),
("What is the value of Golang's community?", "It provides extensive documentation and resources for developers."),
("What is the significance of mastering machine learning basics?", "It's essential for future technological advancements."),
("How does cloud computing improve collaboration?", "By enabling on-demand access to shared resources."),
("What is the purpose of clustering in unsupervised learning?", "To find hidden patterns in data."),
("How does Golang handle concurrency?", "Through goroutines and channels, allowing efficient multitasking."),
("What does PaaS offer to developers?", "A platform to develop, run, and manage applications."),
("What does SaaS offer to users?", "Access to software applications over the web."),
("How does Golang's syntax benefit developers?", "It aids in writing clean and maintainable code."),
("Why is preprocessing important in machine learning?", "To ensure data quality and improve model performance."),
("What does the memo suggest about the future of cloud computing?", "Businesses will increasingly shift to the cloud."),
("What is a key advantage of using Golang?", "Its performance is comparable to C/C++."),
("What is a common use of the scikit-learn library?", "Implementing machine learning algorithms."),
("How does cloud computing affect business transformation?", "It facilitates digital transformation."),
("What is the role of feature extraction in machine learning?", "To enhance model performance by selecting relevant data features."),
("What is the challenge mentioned regarding cloud computing?", "Security concerns."),
("What does the memo indicate about Golang's development process?", "Frameworks like Gin and Echo streamline it."),
("What is the significance of learning machine learning?", "It enables data-driven decision-making."),
("What does the memo highlight as a benefit of Golang's community?", "It provides documentation and resources for new developers.")]

    dataset_name = "Memo Info Query Agent"
    dataset = client.create_dataset(dataset_name=dataset_name, description="QA pairs of Memo Info")
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
        data="Memo Info Query Agent",
        evaluators=[answer_evaluator],
        experiment_prefix="memo-agent-gpt4o" + "-similarity_score",
        num_repetitions=3,
        metadata={"version": "v1.0"},
    )

    print(experiment_results)

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    main()