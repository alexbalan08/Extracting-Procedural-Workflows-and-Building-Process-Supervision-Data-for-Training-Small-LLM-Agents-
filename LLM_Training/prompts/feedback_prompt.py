from typing import List


def format_feedback_user_message(procedure_text: str, issues: List[str], attempt: int) -> str:
    
    issues_block = "\n".join(f"  {i+1}. {issue}" for i, issue in enumerate(issues))
    return (
        f"Extract the workflow from the following procedure text:\n\n{procedure_text}\n\n"
        f"--- FEEDBACK FROM PREVIOUS ATTEMPT (attempt {attempt}) ---\n"
        f"Your last extraction had the following structural issues that must be fixed:\n"
        f"{issues_block}\n\n"
        "Please reason carefully again and produce a corrected extraction."
    )



#this we fall back if there are no issue
#this is the prompt we feed
#we will pass the ystem pormpt also with detailed strcutre
def format_initial_user_message(procedure_text: str) -> str:
    return f"Extract the workflow from the following procedure text:\n\n{procedure_text}"
