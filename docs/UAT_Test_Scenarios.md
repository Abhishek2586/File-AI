# User Acceptance Testing (UAT) Scenarios
## AI File Assistant — Prompt 26

---

## Prerequisites
- App running at `http://localhost:8501`
- At least 2 NIST PDFs uploaded and indexed
- Valid API key in `.env` file

---

## Test Scenarios

| # | Scenario | Precondition | Steps | Expected Result | Status |
|---|---------|-------------|-------|----------------|--------|
| 1 | First-time user loads app | App started | Open http://localhost:8501 | Sidebar and Ask Questions page visible | ☐ |
| 2 | Upload a single PDF | Empty DB | Go to Upload > select one PDF > click Process | Progress bar shown, success message, file count = 1 | ☐ |
| 3 | Upload multiple PDFs simultaneously | Empty DB | Select 3 PDFs at once > Process | All files processed with individual success/fail messages | ☐ |
| 4 | Upload a non-PDF file | Upload page open | Try to select a .txt file | File uploader rejects it (Streamlit blocks non-PDF extensions) | ☐ |
| 5 | Ask a basic question | At least 1 PDF indexed | Ask "What is the NIST Cybersecurity Framework?" | Non-empty answer returned with at least 1 source | ☐ |
| 6 | Ask about five CSF functions | NIST_CSWP_29.pdf indexed | Ask "What are the six functions of the NIST CSF?" | Answer includes: Govern, Identify, Protect, Detect, Respond, Recover | ☐ |
| 7 | Ask about GOVERN function | NIST_CSWP_29.pdf indexed | Ask "Explain the GOVERN function" | Accurate description mentioning policies, roles and risk management | ☐ |
| 8 | Ask about supply chain risks | NIST_IR_8596 indexed | Ask "How does the framework address supply chain risks?" | Relevant answer citing NIST_IR_8596_iprd.pdf | ☐ |
| 9 | Ask about CSF Tiers | NIST_CSWP_29.pdf indexed | Ask "What is the difference between Tier 2 and Tier 3?" | Clear differentiation between Risk Informed and Repeatable tiers | ☐ |
| 10 | Multi-turn follow-up | Previous question answered | Ask follow-up: "Can you elaborate on the first one?" | Coherent follow-up that references the previous answer's topic | ☐ |
| 11 | Clear chat history | Chat with history | Click "Clear Chat" button | Chat area cleared; new question treated as fresh | ☐ |
| 12 | View Document Management | 2+ PDFs indexed | Navigate to Document Management | Both documents listed with their filenames | ☐ |
| 13 | Search documents list | Multiple PDFs indexed | Type partial filename in search box | Filtered list showing only matching files | ☐ |
| 14 | Delete a document | At least 1 PDF indexed | Click Delete on a document | Document removed; DB count reduced | ☐ |
| 15 | Query after deletion | Document just deleted | Ask about a topic from deleted doc | System returns low-confidence or fallback answer | ☐ |
| 16 | Adjust top_k in Settings | Settings page | Change top_k from 5 to 2 > Save | Next answer uses only 2 sources in its citation | ☐ |
| 17 | Change model in Settings | Settings page | Change model > Save > Ask question | Answer is returned without errors | ☐ |
| 18 | View Analytics Dashboard | At least 5 questions asked | Navigate to Analytics | Metrics show correct document count and question count | ☐ |
| 19 | Ask an out-of-scope question | DB has unrelated docs | Ask "What is the best pizza topping?" | System responds it cannot answer based on provided context | ☐ |
| 20 | App restart persistence | ChromaDB has data | Stop streamlit and restart with `streamlit run app.py` | Previous documents still appear in Document Management | ☐ |

---

## NIST PDF-Specific Test Questions

| Question | Expected Source |
|---------|----------------|
| "What are the CSF Functions?" | NIST_CSWP_29.pdf |
| "Explain the GOVERN function" | NIST_CSWP_29.pdf |
| "What is Tier 3 in the CSF?" | NIST_CSWP_29.pdf |
| "How does the framework address supply chain risks?" | NIST_IR_8596_iprd.pdf |
| "What are CSF Organizational Profiles?" | NIST_CSWP_29.pdf |
| "Why is encryption important?" | NIST_SP_500-291r2.pdf |

---

## Pass Criteria
- ✅ Scenarios 1-5 must pass for basic functionality
- ✅ Scenarios 6-10 must pass for accurate Q&A
- ✅ Scenarios 11-17 must pass for UI completeness
- ✅ Scenarios 18-20 are for edge cases and persistence
