---
applyTo: '**'
---
We need to create a FastAPI endpoint where we expect a document (e.g., PDF, DOCX, or TXT) to be uploaded from the frontend. Once the document is uploaded, we need to extract text from it using appropriate methods:

For .txt files, standard Python file handling packages can be used.

For .pdf or complex documents, we will use llama-index to extract the text effectively.

The extracted text will then be sent to the OpenAI LLM (specifically GPT-4o), which will summarize the content and generate a detailed and extended scenario. For example, the input document may contain a sentence like:

"Looks into the dashboard, checks various tabs, and downloads the report in a PDF or Excel format."

In such a case, the extracted text should be passed to GPT-4o, which will reframe it as professional-level development instructions, suitable for a full-stack developer.

The response from GPT-4o will then be passed to another LLM, specifically GPT-4o. GPT will act as a solution architect and generate the following for each identified task:

task: A development-related task derived from the input (e.g., "UI Development").

hours: An estimated time frame (in hours) to complete the task (e.g., "10hrs").

justification: A breakdown of all activities required for the task (e.g., for "UI Development", this might include "UI/UX Designing, Wireframing, Content development in React").

confidence: A randomly generated confidence level between 85% and 95%.

These outputs should be structured as Python dictionaries, one for each task. Example tasks may include:

UI Development

Frontend Integration

Backend Development

AI Implementation

Database Integration

UAT Testing

Deployment

All the dictionaries should be returned in JSON format.

GPT-4o should not return any final summary—only the structured JSON format as described.
```python
The below is azure openai code to use the o4-mini model for chat completions.
# Make sure to install the openai package with `pip install openai`

import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  
    api_version="2025-01-01-preview",
    azure_endpoint="https://xxxxx.openai.azure.com"
)

response = client.chat.completions.create(
    model="o4-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
)

print(response.choices[0].message.content)

The api key, is stored in an environment variable called API Key.

all the details related to the Azure OpenAI service are stored in environment variables.

Format the .env as per the standard format.