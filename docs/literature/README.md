# Literature material

- `Group11_LR_A ... .docx`: reviewed literature review and project outline.
- `Feedback.md`: project-scope feedback.
- `citations.csv`: reference export.

These files support the research rationale. Implementation decisions and reproducible experiment
records live in the other `docs/` directories.

Current implementation note: after CE-CSL was selected as the final dataset, `domain-bounded` is
implemented as dataset-bounded recognition over CE-CSL `Gloss` tokens and sentence references. Any
business-specific scenario is treated as future migration rather than the main experiment.
