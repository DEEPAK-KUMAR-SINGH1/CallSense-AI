from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import csv

# Load API key from .env
load_dotenv()

# Sample transcript
transcript = """
Customer: Hi, I’m having trouble logging into my account.
Agent: I’m sorry to hear that. Can you tell me what happens when you try to log in?
Customer: It keeps saying my password is incorrect, even though I’m sure it’s right.
Agent: Okay, let’s try resetting your password.
Customer: Alright, but I don’t want to wait too long.
Agent: Don’t worry, it should be quick. I’ll send a reset link to your email.
Customer: Thanks, I got it now.
Agent: Great! Did the reset work?
Customer: Yes, I can log in now. That’s a relief.
Agent: Happy to hear that! Anything else I can help you with?
Customer: Actually, yes. I was trying to update my profile picture, but it keeps failing.
Agent: I see. Can you tell me what error message you get?
Customer: It says “Upload failed. Please try again later.”
Agent: Got it. Sometimes clearing the cache helps. Have you tried that?
Customer: Not yet. Let me try.
Customer: Hmm, still not working.
Agent: Alright, I can manually update it for you. Can you send me the picture?
Customer: Sure, sending now.
Agent: Received it. I’ll update it and let you know once done.
Customer: Thanks, I appreciate that.
Agent: Done! Your profile picture is updated.
Customer: Wow, that was fast. Thank you so much!
Agent: You’re welcome! Anything else today?
Customer: No, that’s all. Very happy with your support.
Agent: Glad to hear! Have a great day.
Customer: You too, bye!
"""

# Load Mistral model
model = ChatMistralAI(model_name="mistral-tiny")

# Prompt for structured CSV-like output
prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are a professional customer support analyst.

Your tasks:
1. Read the following conversation transcript line by line.
2. For each line:
   - Keep the exact transcript text.
   - Write a short 1–2 word summary of that line.
   - Label sentiment as Positive, Negative, or Neutral.

Transcript:
{transcript}

Output strictly as rows (NO explanation, NO extra text).  
Use `|` as separator in this format:

"Customer: Hi, I’m having trouble logging into my account." | "Login issue" | "Negative"
"Agent: I’m sorry to hear that. Can you tell me what happens when you try to log in?" | "Asking details" | "Neutral"
...
"""
)

# Make string output parser object
parser = StrOutputParser()

# make chain
chain = prompt | model | parser

# Run model
result = chain.invoke({"transcript": transcript})

# Clean and parse rows
rows = []
for line in result.split("\n"):
    if "|" in line:
        parts = [p.strip().strip('"') for p in line.split("|")]
        if len(parts) == 3:
            rows.append(parts)

# Save to CSV
with open(" call_analysis.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Transcript", "Summary", "Sentiment"])  # header
    writer.writerows(rows)
print(result)

# Display that CSV file Has been saved successfully
print("\n✅ CSV file saved as call_analysis.csv.csv with proper columns")