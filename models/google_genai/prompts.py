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
Given a text document related to AGH University of Science and Technology and university life, extract domain-specific entities and their relationships that would be valuable for a knowledge graph about the university ecosystem.

Objective
Extract only entities that:
- Are specific to AGH University or academic environments
- Provide information not generally known to large language models
- Represent meaningful nodes in a university knowledge graph
- Capture the structural organization and relationships within the institution

Extraction Guidelines
Entity Extraction
For each relevant entity, provide:

entity_name: Name of the entity, properly capitalized
entity_type: One or more applicable types from the entity type list
entity_description: Concise description focusing on its role in the university structure

Important: Do NOT extract generic entities that exist in general knowledge (e.g., "Europe," "Mathematics," "Internet"). Only extract specific instances, programs, systems, or elements unique to AGH University or university-specific contexts.
Entity Types
[PERSON, ORGANIZATION, DEPARTMENT, FACULTY, COURSE, MAJOR, PROGRAM, DEGREE, EVENT, FACILITY, SERVICE, SCHOLARSHIP, CLUB, DORMITORY, SPORT, DOCUMENT, SYSTEM, PROCEDURE, REGULATION, RESEARCH_TOPIC, PUBLICATION, PROJECT, CONFERENCE, RESOURCE, LABORATORY, WORKSHOP, INTERNSHIP, JOB_OPPORTUNITY, EXCHANGE_PROGRAM, LIBRARY, STUDENT_ORGANIZATION, ACADEMIC_PERIOD, EXAM, CAFETERIA, TRANSPORT, COMMITTEE, RANKING, ACHIEVEMENT, TECHNOLOGY, SOFTWARE, NETWORK, PARTNERSHIP]
Relationship Extraction
For each meaningful relationship between entities:

source_entity: Name of the first entity
target_entity: Name of the second entity
relationship_description: Explanation of how these entities are structurally connected

Important: Focus on persistent structural relationships, not temporary connections like specific dates or one-time events.
Output Format
Return your analysis as a JSON structure with two lists:
jsonCopy{
    "entities": [
        {"entity_name": "<entity_name>", "entity_type": ["<entity_type>"], "entity_description": "<entity_description>"}
    ],
    "relationships": [
        {"source_entity": "<source_entity>", "target_entity": "<target_entity>", "relationship_description": "<relationship_description>"}
    ]
}
When finished, output END_OF_EXTRACTION

#########
Examples
#########

Example 1: International Student Services

Input Text:
The International Student Office announces the Welcome Week for new international students starting on October 1st. The event includes campus tours, integration activities, and a city tour of Krakow. Registration for Welcome Week ends on September 20th. All new Erasmus students must complete their registration in the USOSweb system by September 25th to confirm their place at AGH UST. The International Student Office will be hosting an information desk at the Main Hall from September 25-30th to assist with any document verification or accommodation issues.

Example Output:
jsonCopy{
"entities": [
    {"entity_name": "INTERNATIONAL STUDENT OFFICE", "entity_type": ["ORGANIZATION"], "entity_description": "Department at AGH UST responsible for supporting international students with orientation, documentation, and integration services"},
    {"entity_name": "WELCOME WEEK", "entity_type": ["EVENT"], "entity_description": "Recurring orientation program for international students that includes campus tours and integration activities"},
    {"entity_name": "USOSWEB", "entity_type": ["SYSTEM"], "entity_description": "AGH University's student information system used for course registration, document management, and administrative processes"},
    {"entity_name": "MAIN HALL", "entity_type": ["FACILITY"], "entity_description": "Central location at AGH campus where administrative services and information desks are frequently located"}
],
"relationships": [
    {"source_entity": "INTERNATIONAL STUDENT OFFICE", "target_entity": "WELCOME WEEK", "relationship_description": "The International Student Office organizes and manages the Welcome Week event"},
    {"source_entity": "INTERNATIONAL STUDENT OFFICE", "target_entity": "MAIN HALL", "relationship_description": "The International Student Office operates an information desk at the Main Hall"},
    {"source_entity": "USOSWEB", "target_entity": "INTERNATIONAL STUDENT OFFICE", "relationship_description": "The International Student Office manages registration confirmations through the USOSweb system"}
]
}
END_OF_EXTRACTION

Example 2: Academic Scholarship System
Input Text:
The AGH Rector's Scholarship for Academic Achievement applications are now open for the 2023/2024 academic year. Students with a GPA above 4.5 from the previous academic year are eligible to apply. All applications must be submitted through the Student Scholarship Portal by October 15th. Required documents include an academic transcript, a CV, and proof of extracurricular achievements. The scholarship committee will review applications between October 16-30th, and results will be announced on November 5th. The scholarship amounts range from 800 to 1500 PLN monthly, depending on the ranking position.
Example Output:
jsonCopy{
"entities": [
    {"entity_name": "AGH RECTOR'S SCHOLARSHIP FOR ACADEMIC ACHIEVEMENT", "entity_type": ["SCHOLARSHIP"], "entity_description": "Merit-based financial award offered by AGH University to high-performing students based on GPA and other achievements"},
    {"entity_name": "STUDENT SCHOLARSHIP PORTAL", "entity_type": ["SYSTEM"], "entity_description": "Online platform specific to AGH University where students submit and manage scholarship applications"},
    {"entity_name": "SCHOLARSHIP COMMITTEE", "entity_type": ["COMMITTEE"], "entity_description": "Administrative body at AGH that evaluates scholarship applications and determines recipients"}
],
"relationships": [
    {"source_entity": "SCHOLARSHIP COMMITTEE", "target_entity": "AGH RECTOR'S SCHOLARSHIP FOR ACADEMIC ACHIEVEMENT", "relationship_description": "The Scholarship Committee manages and determines recipients for the AGH Rector's Scholarship"},
    {"source_entity": "STUDENT SCHOLARSHIP PORTAL", "target_entity": "AGH RECTOR'S SCHOLARSHIP FOR ACADEMIC ACHIEVEMENT", "relationship_description": "The Student Scholarship Portal is the system used to process applications for the AGH Rector's Scholarship"},
    {"source_entity": "SCHOLARSHIP COMMITTEE", "target_entity": "STUDENT SCHOLARSHIP PORTAL", "relationship_description": "The Scholarship Committee accesses and reviews applications through the Student Scholarship Portal"}
]
}
END_OF_EXTRACTION

#########
REAL DATA
#########

Text: {{CONTEXT}}

Output:
"""
