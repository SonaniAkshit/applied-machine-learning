# main.py

import random
import string
import json
import os
from datetime import datetime

# -------------------------------
# Utility Functions
# -------------------------------

def generate_random_name():
    first = random.choice(["Akshit", "Rahul", "Priya", "Neha", "Amit", "Sara"])
    last = random.choice(["Patel", "Sharma", "Verma", "Gupta", "Mehta"])
    return f"{first} {last}"

def generate_random_email(name):
    domains = ["gmail.com", "yahoo.com", "outlook.com"]
    return name.lower().replace(" ", ".") + "@" + random.choice(domains)

def generate_random_score():
    return random.randint(40, 100)

# -------------------------------
# Data Model
# -------------------------------

class Student:
    def __init__(self, student_id, name, email, scores):
        self.student_id = student_id
        self.name = name
        self.email = email
        self.scores = scores

    def average_score(self):
        return sum(self.scores) / len(self.scores)

    def to_dict(self):
        return {
            "id": self.student_id,
            "name": self.name,
            "email": self.email,
            "scores": self.scores,
            "average": self.average_score()
        }

# -------------------------------
# Data Manager
# -------------------------------

class StudentManager:
    def __init__(self, file_path="students.json"):
        self.file_path = file_path
        self.students = []
        self.load_data()

    def generate_students(self, count=50):
        for i in range(count):
            name = generate_random_name()
            email = generate_random_email(name)
            scores = [generate_random_score() for _ in range(5)]
            student = Student(i + 1, name, email, scores)
            self.students.append(student)

    def save_data(self):
        data = [s.to_dict() for s in self.students]
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)

    def load_data(self):
        if not os.path.exists(self.file_path):
            return

        with open(self.file_path, "r") as f:
            data = json.load(f)
            for item in data:
                student = Student(
                    item["id"],
                    item["name"],
                    item["email"],
                    item["scores"]
                )
                self.students.append(student)

    def display_students(self, limit=10):
        print("\n--- Student List ---")
        for student in self.students[:limit]:
            print(f"{student.student_id} | {student.name} | Avg: {student.average_score():.2f}")

    def top_students(self, n=5):
        sorted_students = sorted(self.students, key=lambda s: s.average_score(), reverse=True)
        print("\n--- Top Students ---")
        for s in sorted_students[:n]:
            print(f"{s.name} -> {s.average_score():.2f}")

    def search_student(self, keyword):
        results = [s for s in self.students if keyword.lower() in s.name.lower()]
        print("\n--- Search Results ---")
        for s in results:
            print(f"{s.student_id} | {s.name} | {s.email}")

# -------------------------------
# Logger
# -------------------------------

class Logger:
    def __init__(self, file_name="app.log"):
        self.file_name = file_name

    def log(self, message):
        with open(self.file_name, "a") as f:
            f.write(f"{datetime.now()} - {message}\n")

# -------------------------------
# CLI Menu
# -------------------------------

def show_menu():
    print("\n====== MENU ======")
    print("1. Generate Students")
    print("2. Save Data")
    print("3. Load Data")
    print("4. Display Students")
    print("5. Top Students")
    print("6. Search Student")
    print("7. Exit")

def main():
    manager = StudentManager()
    logger = Logger()

    while True:
        show_menu()
        choice = input("Enter choice: ")

        if choice == "1":
            count = int(input("How many students? "))
            manager.generate_students(count)
            logger.log(f"Generated {count} students")

        elif choice == "2":
            manager.save_data()
            logger.log("Saved data to file")

        elif choice == "3":
            manager.load_data()
            logger.log("Loaded data from file")

        elif choice == "4":
            manager.display_students()
            logger.log("Displayed students")

        elif choice == "5":
            manager.top_students()
            logger.log("Viewed top students")

        elif choice == "6":
            keyword = input("Enter name to search: ")
            manager.search_student(keyword)
            logger.log(f"Searched for {keyword}")

        elif choice == "7":
            logger.log("Application exited")
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")

# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    main()