from langchain.indexes.prompts.entity_extraction import ENTITY_EXTRACTION_PROMPT

QUERY_AUGMENTATION_PROMPT_TEMPLATE = f"""
You are an AI assistant specializing in academic information 
retrieval for AGH University of Science and Technology.
Given the user's query, generate three alternative phrasings 
that maintain the original intent but vary in wording and structure.
These variations should encompass different perspectives and 
terminologies to capture a broad spectrum of relevant documents.

QUERY: {{QUERY}}

THREE ALTERNATIVE PHRASINGS:
"""


ENHANCE_SEARCH_PROMPT_TEMPLATE = f"""
You are an advanced AI model assisting a Retrieval-Augmented Generation (RAG) system
designed to answer user queries using context retrieved from a database.

Based on the provided context and the user’s query, identify if there is information missing
in the provided context that is necessary to answer the query fully.

If there is missing information:
- Formulate up to three questions that can help retrieve the missing information.
- Based on the available chunks, write a concise summary that includes all the key
 details needed to answer the query. The summary should be comprehensive and cover all 
 important that addresses the query.

If there is no missing information:
- Return empty python dictionary

Format your output as a python dict, "summary" key should contain string with summary,
 "questions" key should contain list of question strings.

 QUERY: {{QUERY}}

 CONTEXT: {{CONTEXT}}  
"""


ANSWER_GENERATION_PROMPT_TEMPLATE = f"""
You are a knowledgeable and reliable assistant.
Your task is to answer the user’s question using only the information provided in the source documents below.
Do not add any information that is not present in the documents.
If the necessary answer is not found within the data, respond with "I'm not able to find the answer for your question."
 or ask a clarifying question if needed.

You have to detect the users questions language and answer ONLY in this langauge.

Your answer should be comprehensive.

Source Documents:
{{CONTEXT}}

User Question:
{{QUERY}}

Answer:
"""


GRAPH_EXTRACTION_PROMPT = """
Given a text document that is potentially relevant to AGH University of Science and Technology and university life, identify all entities of the specified types from the text and all relationships among the identified entities.
-Steps-

Identify all entities. For each identified entity, extract the following information:


entity_name: Name of the entity, capitalized
entity_type: One of the following types: [PERSON, ORGANIZATION, DEPARTMENT, FACULTY, COURSE, MAJOR, PROGRAM, DEGREE, EVENT, FACILITY, SERVICE, DEADLINE, SCHOLARSHIP, CLUB, DORMITORY, SPORT, DOCUMENT, SYSTEM, PROCEDURE, REGULATION, RESEARCH_TOPIC, PUBLICATION, PROJECT, CONFERENCE, RESOURCE, LABORATORY, WORKSHOP, INTERNSHIP, JOB_OPPORTUNITY, EXCHANGE_PROGRAM, LIBRARY, STUDENT_ORGANIZATION, ACADEMIC_PERIOD, EXAM, CAFETERIA, OFFICE_HOURS, TRANSPORT, COMMITTEE, RANKING, ACHIEVEMENT, TECHNOLOGY, SOFTWARE, NETWORK, PARTNERSHIP]
entity_description: Comprehensive description of the entity's attributes and activities
Format each entity using Python dictionary syntax as follows:
{"entity_name": "<entity_name>", "entity_type": ["<entity_type>"], "entity_description": "<entity_description>"}


From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other.
For each pair of related entities, extract the following information:


source_entity: name of the source entity, as identified in step 1
target_entity: name of the target entity, as identified in step 1
relationship_description: explanation as to why you think the source entity and the target entity are related to each other
Format each relationship using Python dictionary syntax as follows:
{"source_entity": "<source_entity>", "target_entity": "<target_entity>", "relationship_description": "<relationship_description>"}


Return output in English as a JSON structure with two lists:

pythonCopy{
    "entities": [
        # List of entity dictionaries
    ],
    "relationships": [
        # List of relationship dictionaries
    ]
}

When finished, output END_OF_EXTRACTION

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,EVENT,DEADLINE
Text:
The International Student Office announces the Welcome Week for new international students starting on October 1st. The event includes campus tours, integration activities, and a city tour of Krakow. Registration for Welcome Week ends on September 20th. All new Erasmus students must complete their registration in the USOSweb system by September 25th to confirm their place at AGH UST. The International Student Office will be hosting an information desk at the Main Hall from September 25-30th to assist with any document verification or accommodation issues.
######################
Output:
{
"entities": [
{"entity_name": "INTERNATIONAL STUDENT OFFICE", "entity_type": ["ORGANIZATION"], "entity_description": "Office responsible for international student affairs at AGH UST, organizing Welcome Week and providing information"},
{"entity_name": "WELCOME WEEK", "entity_type": ["EVENT"], "entity_description": "Orientation event for new international students starting on October 1st with campus tours, integration activities, and a city tour"},
{"entity_name": "WELCOME WEEK REGISTRATION", "entity_type": ["DEADLINE"], "entity_description": "Registration deadline for Welcome Week is September 20th"},
{"entity_name": "USOSWEB REGISTRATION", "entity_type": ["DEADLINE"], "entity_description": "Erasmus students must register in USOSweb by September 25th to confirm their place"},
{"entity_name": "USOSWEB", "entity_type": ["SYSTEM"], "entity_description": "University system where students must complete their registration"},
],
"relationships": [
{"source_entity": "INTERNATIONAL STUDENT OFFICE", "target_entity": "WELCOME WEEK", "relationship_description": "The International Student Office organizes the Welcome Week event"},
{"source_entity": "WELCOME WEEK", "target_entity": "WELCOME WEEK REGISTRATION", "relationship_description": "Students must register for Welcome Week by the registration deadline"},
{"source_entity": "INTERNATIONAL STUDENT OFFICE", "target_entity": "INFORMATION DESK", "relationship_description": "The International Student Office will be hosting the information desk"},
{"source_entity": "INFORMATION DESK", "target_entity": "MAIN HALL", "relationship_description": "The information desk will be located at the Main Hall"},
{"source_entity": "USOSWEB", "target_entity": "USOSWEB REGISTRATION", "relationship_description": "USOSweb is the system where students must register by the deadline"}
]
}
END_OF_EXTRACTION
######################
Example 2:
Entity_types: SCHOLARSHIP,DEADLINE,PROCEDURE
Text:
The AGH Rector's Scholarship for Academic Achievement applications are now open for the 2023/2024 academic year. Students with a GPA above 4.5 from the previous academic year are eligible to apply. All applications must be submitted through the Student Scholarship Portal by October 15th. Required documents include an academic transcript, a CV, and proof of extracurricular achievements. The scholarship committee will review applications between October 16-30th, and results will be announced on November 5th. The scholarship amounts range from 800 to 1500 PLN monthly, depending on the ranking position.
######################
Output:
{
"entities": [
{"entity_name": "AGH RECTOR'S SCHOLARSHIP FOR ACADEMIC ACHIEVEMENT", "entity_type": ["SCHOLARSHIP"], "entity_description": "Scholarship offered to students with GPA above 4.5 with amounts ranging from 800-1500 PLN monthly"},
{"entity_name": "STUDENT SCHOLARSHIP PORTAL", "entity_type": ["SYSTEM"], "entity_description": "Online portal where scholarship applications must be submitted"},
{"entity_name": "APPLICATION SUBMISSION", "entity_type": ["DEADLINE"], "entity_description": "Scholarship applications must be submitted by October 15th"},
{"entity_name": "SCHOLARSHIP APPLICATION REVIEW", "entity_type": ["PROCEDURE"], "entity_description": "Review process conducted by the scholarship committee between October 16-30th"},
{"entity_name": "APPLICATION REQUIREMENTS", "entity_type": ["PROCEDURE"], "entity_description": "Required documents include academic transcript, CV, and proof of extracurricular achievements"},
{"entity_name": "SCHOLARSHIP COMMITTEE", "entity_type": ["ORGANIZATION", "COMMITTEE"], "entity_description": "Committee that reviews scholarship applications"}
],
"relationships": [
{"source_entity": "AGH RECTOR'S SCHOLARSHIP FOR ACADEMIC ACHIEVEMENT", "target_entity": "APPLICATION SUBMISSION", "relationship_description": "Applications for this scholarship must be submitted by this deadline"},
{"source_entity": "AGH RECTOR'S SCHOLARSHIP FOR ACADEMIC ACHIEVEMENT", "target_entity": "STUDENT SCHOLARSHIP PORTAL", "relationship_description": "Applications for this scholarship must be submitted through this portal"},
{"source_entity": "AGH RECTOR'S SCHOLARSHIP FOR ACADEMIC ACHIEVEMENT", "target_entity": "APPLICATION REQUIREMENTS", "relationship_description": "These requirements must be met to apply for the scholarship"},
{"source_entity": "SCHOLARSHIP COMMITTEE", "target_entity": "SCHOLARSHIP APPLICATION REVIEW", "relationship_description": "The committee conducts this review process"},
{"source_entity": "SCHOLARSHIP APPLICATION REVIEW", "target_entity": "RESULTS ANNOUNCEMENT", "relationship_description": "The review process leads to results being announced on this date"}
]
}
END_OF_EXTRACTION
######################

######################
-Real Data-
######################
Text: {{CONTEXT}}
######################
Output:
"""
