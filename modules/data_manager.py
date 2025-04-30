import pandas as pd
import random
import streamlit as st

class QuestionManager:
    def __init__(self, file_path):
        self.questions_df = pd.read_csv(file_path)
        self.current_index = 0
        self.shuffle_questions()
        
    def shuffle_questions(self):
        self.questions_df = self.questions_df.sample(frac=1).reset_index(drop=True)
        
    def get_current_question(self):
        if self.current_index < len(self.questions_df):
            return self.questions_df.iloc[self.current_index]
        return None
        
    def next_question(self):
        self.current_index += 1
        return self.get_current_question()
        
    def reset_session(self):
        self.current_index = 0
        self.shuffle_questions()

    def get_progress(self):
        total = len(self.questions_df)
        current = self.current_index + 1
        return current, total, current / total 