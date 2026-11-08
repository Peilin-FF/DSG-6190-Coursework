"""Prompt templates.

The DSG grounding and final-answer prompts are copied from the authors' released notebook.
The CCoT prompts follow Mitra et al. (CVPR 2024). Every method receives the concept name.
"""

GROUND_SYSTEM = (
    "You are an accurate question answering engine that uses the given imagined scenario to "
    "associate abstract concepts with actual objects in the image. Please disregard grammar that may "
    "be inaccurate in the question. By answering correctly, you are giving helpful grounded contexts "
    "about the image for further question answering."
)

FINAL_REMINDER = (
    "The stated imagined scenario in the text may not always be correct; it serves to guide reasoning. "
    "Disregard any inaccurate grammar. Think of the imagined scenarios as helpful contexts for what "
    "concepts may align with which objects in the image."
)

GROUND_QUESTION = "Imagine that the image represents {concept}{conditions} what is the {term}?"
GROUND_CONDITION = " and the {c} is {v},"
GROUND_SUFFIX = {
    "phrase": " Answer with one word or phrase only, do not elaborate.",
    "sentence": " Answer with one short sentence that names the actual objects in the image.",
}

FINAL_CONTEXT = "Imagine that the image represents {concept}{conditions}."
FINAL_CONDITION = " and the {c} is {v}"
FINAL_SUFFIX = "Answer concisely without explanation, with the final answer only."

COT_SUFFIX = ("Let's think step by step. Then give the final answer on the last line in the form "
              "'Answer: <answer>'.")

CCOT_SCENE_GRAPH = (
    "For the provided image and its associated question, generate a scene graph in JSON format that "
    "includes the following:\n"
    "1. Objects that are relevant to answering the question\n"
    "2. Object attributes that are relevant to answering the question\n"
    "3. Object relationships that are relevant to answering the question\n\n"
    "Just generate the scene graph in JSON format. Do not say extra words.\n\n"
    "Question: {question}"
)
CCOT_ANSWER = "Scene Graph: {scene_graph}\n\nUse the image and scene graph as context and answer the following question: {question}"

SCHEMA_COT = (
    "The abstract concept {concept} can be described by the following schema, a program in which "
    "p(x | y) means that component x is interpreted given y:\n{program}\n"
    "First state, for each component of the schema, what it corresponds to in the image. "
    "Then answer the question.\n{question}\n"
    "Give the final answer on the last line in the form 'Answer: <answer>'."
)

# Concept-agnostic schema used as a control (it names no concept-specific component).
GENERIC_PROGRAM = (
    "p(image | concept={concept}) =\n"
    "  p(objects | concept={concept})\n"
    "  p(attributes | concept={concept}, objects)\n"
    "  p(relations | concept={concept}, objects)\n"
    "  p(image | objects, attributes, relations)"
)

# Schema extraction (LLM), Appendix C.1 of the paper. The default example has 3 components; the
# complexity study varies the size of the example (3, 5, 7, 9 components, Appendix A.5).
SCHEMA_EXAMPLES = {
    3: ["gen(faculty | concept=academia)",
        "gen(students | concept=academia)",
        "gen(research-output | concept=academia, faculty, students)"],
    5: ["gen(faculty | concept=academia)",
        "gen(students | concept=academia)",
        "gen(courses | concept=academia, faculty, students)",
        "gen(research-output | concept=academia, faculty, students)",
        "gen(publications | concept=academia, research-output)"],
    7: ["gen(campus | concept=academia)",
        "gen(faculty | concept=academia, campus)",
        "gen(students | concept=academia, campus)",
        "gen(courses | concept=academia, faculty, students)",
        "gen(funding | concept=academia, faculty)",
        "gen(research-output | concept=academia, faculty, students, funding)",
        "gen(publications | concept=academia, research-output)"],
    9: ["gen(campus | concept=academia)",
        "gen(faculty | concept=academia, campus)",
        "gen(students | concept=academia, campus)",
        "gen(courses | concept=academia, faculty, students)",
        "gen(degrees | concept=academia, students, courses)",
        "gen(funding | concept=academia, faculty)",
        "gen(research-output | concept=academia, faculty, students, funding)",
        "gen(publications | concept=academia, research-output)",
        "gen(conferences | concept=academia, publications)"],
}
SCHEMA_LIMIT_WORDS = {3: "four", 5: "six", 7: "eight", 9: "ten"}
SCHEMA_REQUEST = (
    "Can you give me a program representing the schema for a concept? For example,\n"
    "gen(concept=academia) =\n{example}\n"
    "Please do the same for gen(concept={concept}) in the same format without explanation. "
    "Keep the program simple with {limit} or less components. "
    "Use only the most necessary parts of the schema that can be mapped to objects in an image."
)


def schema_request(concept: str, size: int = 3) -> str:
    return SCHEMA_REQUEST.format(example="\n".join(SCHEMA_EXAMPLES[size]), concept=concept.replace(" ", "-"),
                                 limit=SCHEMA_LIMIT_WORDS[size])


# DSG-Q (our extension): question-specific components appended to the concept schema.
QSCHEMA_REQUEST = (
    "Here is a program representing the schema for the concept {concept}:\n{program}\n"
    "A question about an image of {concept} is: \"{question}\"\n"
    "Add at most two new components that must be identified in the image to answer this question and that "
    "the program does not contain yet. Write only the new lines in the same format, for example "
    "gen(new-component | concept={concept_tok}, existing-component), without explanation. "
    "If the program already contains everything that the question needs, write NONE."
)

# DSG-X (our extension): role-exclusive grounding. A later component may not reuse an earlier grounding.
EXCLUSIVE_CLAUSE = " It is not the same as {items}."
