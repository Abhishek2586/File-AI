# User Guide — AI File Assistant
## Getting the Most from Your Document Assistant

---

## 🚀 Getting Started

### Step 1: Start the Application
```bash
cd d:\Downloads\Infosys\ai-file-assistant
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

### Step 2: Upload Your Documents
1. Click **"Upload Documents"** in the left sidebar
2. Click "Browse files" and select one or more PDF files
3. Click **"Process Documents"** — a progress bar will appear
4. Wait for the success message — your documents are now indexed!

### Step 3: Ask Questions
1. Click **"Ask Questions"** in the left sidebar
2. Type your question in the text box at the bottom of the screen
3. Press Enter or click the send button
4. Read the answer and expand **"📚 Sources & Citations"** to see which document and page the answer came from

---

## 📤 How to Upload Documents

**Supported format:** PDF only (`.pdf` files)

**Best practices:**
- Use text-based PDFs — scanned documents without OCR may extract poorly
- For best accuracy, upload all related documents at once
- Uploading the same document twice is safe — duplicates are automatically detected

**What happens when you upload:**
1. Text is extracted from each PDF page
2. Text is cleaned and split into ~500-word chunks
3. Each chunk is converted into a mathematical vector (embedding)
4. Vectors are stored in your local ChromaDB database

---

## 💬 How to Ask Effective Questions

**Good questions are:**
- Specific: *"What are the six core functions of the NIST CSF?"*
- Direct: *"What does the Protect function cover?"*
- Topic-focused: *"How does the framework address supply chain risks?"*

**Less effective questions:**
- Vague: *"Tell me about security"*
- Not in the documents: *"What is today's weather?"*

**Tips:**
- If your answer is missing context, rephrase the question with more keywords
- Use follow-up questions to drill down: after asking about "NIST CSF", ask "Can you explain the Govern function in more detail?"
- The assistant remembers the last 3 turns of conversation — use this for follow-up questions

---

## 📄 Managing Your Documents

Navigate to **Document Management** to:
- See all currently indexed PDFs
- Search for a document by name
- Delete a document from the knowledge base (this cannot be undone, but you can re-upload)

---

## ⚙️ Adjusting Settings

The **Settings** page lets you configure:

| Setting | Description | Recommended Value |
|---------|-------------|------------------|
| Model | Which AI model to use for answers | `gpt-3.5-turbo` for speed, `gpt-4o` for best accuracy |
| Temperature | How creative vs. factual the answers are | 0.2–0.4 for document Q&A |
| Max Tokens | Maximum answer length | 400–800 tokens |
| Top K Sources | How many document chunks to retrieve | 3–7 |

Always click **"Save Settings"** after making changes.

---

## 📊 Analytics Dashboard

The Analytics page shows:
- How many documents are in the knowledge base
- How many questions you've asked in this session
- Average response time and confidence score
- Chart: documents indexed

---

## ❓ FAQ

**Q: What types of PDF are supported?**  
A: Any text-based PDF. Scanned image PDFs need OCR pre-processing before uploading.

**Q: How accurate are the answers?**  
A: The assistant uses semantic search to find the most relevant text, then asks an LLM to summarize. Accuracy depends heavily on whether the PDF contains the answer. The confidence score (shown under citations) gives an estimate.

**Q: What happens to my documents?**  
A: All documents and embeddings are stored **locally** on your machine. Nothing is uploaded to any external server except for small text chunks sent to the LLM API to generate answers.

**Q: How much does it cost to use?**  
A: Costs depend on your API provider. With the FastRouter API and `gpt-3.5-turbo`, each question typically costs $0.001–$0.003. Uploading a 50-page PDF costs approximately $0.02–$0.05 in embedding API calls.

**Q: My answer says "I cannot find this in the provided context." What does that mean?**  
A: The semantic search did not find relevant chunks in your uploaded documents. Try uploading more documents or rephrasing your question.

**Q: Can I use this with non-English documents?**  
A: The assistant works best with English PDFs. Non-English documents may produce lower accuracy answers.

---

## 🐛 Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Failed to initialize vector database" | ChromaDB path issue | Check that `data/chroma` directory exists |
| Blank answers | API key expired or invalid | Update `.env` file with a valid key |
| Very slow uploads | Large PDF or slow API | Break large PDFs (200+ pages) into smaller files |
| Settings not applying | Forgot to click Save | Click the **Save Settings** button |
