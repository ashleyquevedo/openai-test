import json
import requests
import logging
from tenacity import retry, wait_random_exponential, stop_after_attempt
from test_resumes import TEST_RESUME_1

# temp constant
# set up env file
OPENAI_KEY = "INSERT KEY HERE"

GPT_MODEL = "gpt-3.5-turbo-0613"

# top-level summary description and instructions for AI response
OUTPUT_PARAM_SUMMARY = "Given a person's resume, extract an object with the following keys: skills, education, contact, projects, objective, misc. Each key must be included in the output object, even if the associated array is empty."

# descriptions of resume categories expected to be parsed in AI response object
SKILLS_DESC = "an array of technical skills, including programming languages, libraries, and frameworks, that are related to software development."
EDUCATION_DESC = "an array of information from the resume that relates to educational experience, including degrees and certifications receieved."
EXPERIENCE_DESC = "an array of information from the resume that relates to work or professional experience, including job titles and descriptions."
CONTACT_DESC = "an array of information from the resume that relates to the writer's contact information, including email address, phone number, and portfolio websites."
PROJECTS_DESC = "an array of information from the resume that relates to technical or personal projects."
OBJECTIVE_DESC = (
    "an array of information from the resume that relates to the writer's objective."
)
MISC_DESC = "an array of information from the resume that does not fit into the skills, education, experience, contact, projects, or objective categories. All other information in the resume that has not been categorized should be included here."


# class init includes optional variables to help define functions passed to AI conversation
# .parse method takes a string, which should be derived from a resume, and returns a JSON
class OpenAIResumeParser:
    def __init__(
        self,
        openai_key=OPENAI_KEY,
        gpt_model=GPT_MODEL,
        output_param_summary=OUTPUT_PARAM_SUMMARY,
        skills_desc=SKILLS_DESC,
        education_desc=EDUCATION_DESC,
        experience_desc=EXPERIENCE_DESC,
        contact_desc=CONTACT_DESC,
        projects_desc=PROJECTS_DESC,
        objective_desc=OBJECTIVE_DESC,
        misc_desc=MISC_DESC,
    ):
        self.openai_key = openai_key
        self.gpt_model = gpt_model
        self.output_param_summary = output_param_summary
        self.skills_desc = skills_desc
        self.education_desc = education_desc
        self.experience_desc = experience_desc
        self.contact_desc = contact_desc
        self.projects_desc = projects_desc
        self.objective_desc = objective_desc
        self.misc_desc = misc_desc

        self.output_params = [
            {
                "name": "extract_query_attributes",
                "description": self.output_param_summary,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skills": {
                            "type": "array",
                            "description": self.skills_desc,
                            "items": {
                                "type": "string",
                            },
                        },
                        "education": {
                            "type": "array",
                            "description": self.education_desc,
                            "items": {
                                "type": "string",
                            },
                        },
                        "experience": {
                            "type": "array",
                            "description": self.experience_desc,
                            "items": {
                                "type": "string",
                            },
                        },
                        "contact": {
                            "type": "array",
                            "description": self.contact_desc,
                            "items": {
                                "type": "string",
                            },
                        },
                        "projects": {
                            "type": "array",
                            "description": self.projects_desc,
                            "items": {
                                "type": "string",
                            },
                        },
                        "objective": {
                            "type": "array",
                            "description": self.objective_desc,
                            "items": {
                                "type": "string",
                            },
                        },
                        "misc": {
                            "type": "array",
                            "description": self.misc_desc,
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                },
            }
        ]

    # decorator for retrying after failure
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))

    # utility function for the AI conversation
    def chat_completion_request(self, messages, functions=None, function_call=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.openai_key,
        }
        json_data = {"model": self.gpt_model, "messages": messages}
        if functions is not None:
            json_data.update({"functions": functions})
        if function_call is not None:
            json_data.update({"function_call": function_call})
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

    # returns object parsed according to multiplAI user profile db schema
    def parse(self, resume_str):
        if not isinstance(resume_str, str):
            logging.error(
                f"Argument passed is not a string. See resume_str: {resume_str}"
            )

        # array of messages to pass to AI conversation
        messages = []
        # system-level instructions for AI conversation
        messages.append(
            {
                "role": "system",
                "content": "Don't make assumptions about what values to plug into functions.",
            }
        )
        # pass resume content to AI conversation for parsing
        messages.append(
            {
                "role": "user",
                "content": resume_str,
            }
        )

        chat_response = self.chat_completion_request(
            messages, functions=self.output_params
        )

        try:
            parsed_resume_json = chat_response.json()
            # accessing response from output object with json.loads
            return json.loads(
                # using json.dumps to ensure output is correctly formatted as a json
                json.dumps(
                    parsed_resume_json["choices"][0]["message"]["function_call"][
                        "arguments"
                    ]
                )
            )
        except:
            logging.error("Failed to parse AI conversation response as a JSON.")


parser = OpenAIResumeParser()
print(parser.parse(TEST_RESUME_1))
