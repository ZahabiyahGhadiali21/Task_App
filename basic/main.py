import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import sys

# File path for saving tasks
TASKS_FILE = 'tasks.csv'

# Initialize an empty task list or load existing tasks from CSV
try:
    tasks = pd.read_csv(TASKS_FILE)
except FileNotFoundError:
    tasks = pd.DataFrame(columns=['description', 'priority'])

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv(TASKS_FILE, index=False)

# Train the task priority classifier
vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)
model.fit(tasks['description'], tasks['priority'])

# Function to add a task to the list
def add_task(description, priority):
    global tasks
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()

# Function to remove a task by description
def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

# Function to prioritize a task
def prioritize_task(description, priority):
    global tasks
    tasks.loc[tasks['description'] == description, 'priority'] = priority
    save_tasks()

# Function to recommend a task based on machine learning
# Because of global keyword task will not be removed after being recommended

def recommend_task():
    global tasks
    
    print("Tasks DataFrame:")
    print(tasks)
    
    high_priority_tasks = tasks[tasks['priority'] == 'High']
    print("High Priority Tasks:")
    print(high_priority_tasks)
    
    if high_priority_tasks.empty:
        print("No high-priority tasks available for recommendation.")
    else:
        random_index = random.choice(high_priority_tasks.index)
        task_description = tasks.loc[random_index, 'description']
        print(f"Recommended task: {task_description} - Priority: High")



# Main menu
def main_menu():
    while True:
        print("\nTask Management App")
        print("1. Add Task")
        print("2. Remove Task")
        print("3. List Tasks")
        print("4. Prioritize Task")
        print("5. Recommend Task")
        print("6. Exit")

        choice = input("Select an option: ")

        if choice == "1":
            description = input("Enter task description: ")
            priority = input("Enter task priority (Low/Medium/High): ").capitalize()
            add_task(description, priority)
            print("Task added successfully.")

        elif choice == "2":
            description = input("Enter task description to remove: ")
            remove_task(description)
            print("Task removed successfully.")

        elif choice == "3":
            list_tasks()

        elif choice == "4":
            description = input("Enter task description to prioritize: ")
            priority = input("Enter new priority (Low/Medium/High): ").capitalize()
            prioritize_task(description, priority)
            print("Task prioritized successfully.")

        elif choice == "5":
            if tasks.empty:
                print("No tasks available for recommendation.")
            else:
                recommend_task()

        elif choice == "6":
            print("Goodbye!")
            sys.exit()

        else:
            print("Invalid option. Please select a valid option.")

if __name__ == "__main__":
    main_menu()
