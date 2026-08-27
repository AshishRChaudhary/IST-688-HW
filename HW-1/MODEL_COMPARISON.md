# Model comparison

Document: IST 688 syllabus (pdf)
Question: "Is this course hard?"

## Per-model notes

**gpt-3.5-turbo** --> Short and generic. Gives a one-paragraph "it depends on your background" answer without pulling any specifics from the syllabus.

**gpt-4.1** --> Detailed and well-structured: bold headers, numbered sections, and a summary table, all grounded in actual syllabus numbers (grading breakdown, API cost estimate). Felt comparatively faster to respond and a touch more organized than gpt-5-nano.

**gpt-5-nano** --> Similar depth and syllabus-grounding as gpt-4.1, but plainer formatting (no table) and noticeably slower to respond.

**gpt-5-chat-latest** --> Errored: `404 model_not_found`. OpenAI deprecated this model on 2026-07-23 (recommended successor: `gpt-5.6-sol`), so it isn't testable as written in the assignment.

## cost and speed?

I didn't have access to the OpenAI usage dashboard to see actual token cost, so this is eyeballed rather than measured. gpt-4.1's heavier formatting and more detailed subsections suggest it likely used more output tokens (and so costs more) than gpt-5-nano's plainer response. That said, gpt-4.1 also felt faster and a bit more structured in practice than gpt-5-nano. gpt-3.5-turbo seems cheapest.

## Which answer is best?

For me gpt-4.1 and gpt-5-nano are both clearly better than gpt-3.5-turbo, they actually read the syllabus and answer with real numbers instead of a generic answer. Between the two, gpt-4.1 was ahead slightly on polish and readability (formatting and summary table).