# DesiCrew assessment brief

Source: supplied Questions.docx assessment brief, received 7 October 2026.

The email requires completed solutions, source code, and screenshots or a screen recording of the working solution. GitHub link optional but recommended. Questions 1 and 3 must use the supplied sample files. Deadline written as 9 October, 9:00 AM; the email does not explicitly specify the deadline timezone.

1. Attached is an excel sheet with sample inventory data. Build an agent that answers questions about a provided excel dataset. The agent must be able to: write and execute code to query the data, look up definitions or context using a search tool, and summarise findings in plain English. The user should be able to interact with it via a simple chat interface.
2. Using your own documents as a knowledge base, build a document-aware support assistant that maintains context across a multi-turn conversation. The assistant should: remember what the user has already asked within the session, avoid repeating information it has already provided, gracefully handle topic switches, and cite the specific document section it is drawing from in each response. Demonstrate it across a conversation of at least 10 turns.
3. You are provided with a set of 10 mixed documents spanning identity proofs and insurance proposal forms. Build a pipeline that: (1) classifies each document by type, (2) extracts the specific fields listed below for each document type, and (3) flags any document or field it cannot extract with high confidence for human review. Pay particular attention to handwritten fields — these are the primary test of your pipeline's robustness.
Required extraction targets by document type:
	Document Type
	Fields to Extract

	Aadhaar Card
	Aadhaar Number, Full Name, Date of Birth, Address

	PAN Card
	PAN Number, Full Name, Father's Name, Date of Birth

	Driving Licence
	DL Number, Name, Date of Issue, Valid Till date

	Passport
	Passport Number, Date of Birth, Date of Expiry, MRZ Line 2

	NACH / ECS Mandate (handwritten)
	Bank Account Number, IFSC Code, Bank Name, Amount (figures), Frequency

	FATCA Annexure Form (handwritten)
	Policy Number, TIN / PAN, Father's Name, Place of Birth, Nationality

	Benefit Illustration Declaration (handwritten)
	Application Number, Policyholder Name, Date, Place

	Moral Hazard Questionnaire (handwritten)
	Application Number, Name of Life Assured, Nominee Relationship, Date, Place

	Multiple Policies Consent Form (handwritten)
	Proposer Name, Reason for Multiple Policies (selected checkbox), Date, Place

	Suitability Profiler Declaration (handwritten)
	Application Number, Name of Life Assured, Name of Agent/SP, Date, Place


Deliverables:
· Extraction output as structured JSON for each document
· A confidence score per field (not just per document)
· A flagging report listing fields below your confidence threshold, with the threshold value you chose and your rationale for it
· A brief note on how you handled handwritten text differently from printed text, and any failure cases you observed
Evaluation criteria: Field-level accuracy will be assessed against a ground truth answer key. Particular weight will be given to correct extraction of handwritten fields — IFSC, TIN, bank account numbers, dates, and place names — as these are the highest-value and hardest targets in a real insurance onboarding pipeline.
