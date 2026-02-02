from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import threading
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")

# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_FOLDER = "uploaded_pdfs"
INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = {'pdf'}

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size


# ============================================================================
# GLOBAL VARIABLES (Thread-safe)
# ============================================================================

vectorstore = None
retriever = None
embeddings = None
index_lock = threading.Lock()  # Thread safety


# ============================================================================
# INITIALIZE SYSTEM
# ============================================================================

def initialize_embeddings():
    """Load embedding model once."""
    global embeddings
    
    if embeddings is None:
        print("🔧 Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embeddings loaded")
    
    return embeddings


def load_or_create_index():
    """Load existing index or create empty one."""
    global vectorstore, retriever
    
    with index_lock:
        emb = initialize_embeddings()
        
        # Try loading existing index
        index_file = os.path.join(INDEX_PATH, "index.faiss")
        
        if os.path.exists(index_file):
            try:
                print(f"📂 Loading existing index from {INDEX_PATH}...")
                vectorstore = FAISS.load_local(
                    INDEX_PATH,
                    emb,
                    allow_dangerous_deserialization=True
                )
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                print(f"✅ Index loaded: {vectorstore.index.ntotal} vectors")
                return True
                
            except Exception as e:
                print(f"⚠️  Error loading index: {e}")
                print("   Creating new index...")
        
        # Create empty index
        print("🔧 Creating new empty index...")
        from langchain.schema import Document
        
        dummy_doc = Document(
            page_content="Medical chatbot initialized. Upload PDFs to add knowledge.",
            metadata={"source": "system", "type": "initialization"}
        )
        
        vectorstore = FAISS.from_documents([dummy_doc], emb)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Save initial index
        vectorstore.save_local(INDEX_PATH)
        print("✅ New index created")
        
        return True


def add_pdf_to_index(pdf_path):
    """
    Add a PDF to the vector store.
    Returns: (success, message, num_chunks_added)
    """
    global vectorstore, retriever
    
    try:
        with index_lock:
            print(f"\n📄 Processing: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"   ✅ Loaded {len(docs)} pages")
            
            if not docs:
                return False, "PDF is empty or unreadable", 0
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)
            print(f"   ✅ Created {len(chunks)} chunks")
            
            # Add to vectorstore
            old_count = vectorstore.index.ntotal
            vectorstore.add_documents(chunks)
            new_count = vectorstore.index.ntotal
            
            # Update retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Save updated index
            vectorstore.save_local(INDEX_PATH)
            print(f"   ✅ Index updated: {old_count} → {new_count} vectors")
            
            return True, f"Added {len(chunks)} chunks to index", len(chunks)
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, str(e), 0


# ============================================================================
# LLM SETUP
# ============================================================================

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile",
    temperature=0.3
)


# System prompt with RAG
rag_prompt_template = """You are a medical information assistant with access to uploaded medical documents.

RETRIEVED CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{question}

🚨 CRITICAL RULES:
1. EMERGENCIES: Chest pain, breathing difficulty, severe bleeding 
   → "⚠️ MEDICAL EMERGENCY! Call 911/108 IMMEDIATELY!"

2. BOUNDARIES: You CANNOT diagnose or prescribe
   → Always say "Consult a doctor for diagnosis/prescription"

3. ICD CODES: E11.9=Type 2 Diabetes, I10=Hypertension, J45.9=Asthma, etc.

4. MISINFORMATION: Start with "❌ NO" when correcting false claims

5. LANGUAGE: Support English and Hindi

RESPONSE GUIDELINES:
- Use CONTEXT if relevant
- If context not helpful, use general medical knowledge
- Be concise (<150 words)
- Always recommend doctor consultation
- Cite documents when using their info

Your answer:"""


def get_bot_response(user_question):
    """Get chatbot response with RAG."""
    
    try:
        # Retrieve context
        with index_lock:
            if retriever and vectorstore.index.ntotal > 1:
                # ⬇️ YE LINE CHANGE KARO
                docs = retriever.invoke(user_question)  # PEHLE get_relevant_documents tha
                
                context = "\n\n".join([
                    f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                    for doc in docs
                ])
            else:
                context = "No documents uploaded yet."
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", rag_prompt_template)
        ])
        
        # Generate response
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": user_question
        })
        
        return response
        
    except Exception as e:
        print(f"❌ Error in get_bot_response: {e}")
        return "Sorry, I encountered an error. Please try again."


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_index_stats():
    """Get current index statistics."""
    with index_lock:
        if vectorstore:
            num_vectors = vectorstore.index.ntotal
        else:
            num_vectors = 0
    
    uploaded_pdfs = list(Path(UPLOAD_FOLDER).glob("*.pdf"))
    
    return {
        "num_vectors": num_vectors,
        "num_pdfs": len(uploaded_pdfs),
        "pdfs": [p.name for p in uploaded_pdfs]
    }


# ============================================================================
# ROUTES - CHATBOT
# ============================================================================

@app.route("/")
def home():
    """Main chatbot interface."""
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chat():
    """Chat endpoint."""
    user_msg = request.form.get("msg", "").strip()
    
    if not user_msg:
        return "Please enter a message", 400
    
    try:
        reply = get_bot_response(user_msg)
        return reply
    except Exception as e:
        return f"Error: {str(e)}", 500


# ============================================================================
# ROUTES - ADMIN PANEL
# ============================================================================

@app.route("/admin")
def admin():
    """Admin panel for managing PDFs."""
    stats = get_index_stats()
    return render_template("admin.html", stats=stats)


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF upload and add to index."""
    
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('admin'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('admin'))
    
    if not allowed_file(file.filename):
        flash('Only PDF files allowed', 'error')
        return redirect(url_for('admin'))
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Add to index
        success, message, num_chunks = add_pdf_to_index(filepath)
        
        if success:
            flash(f'✅ Successfully added "{filename}" ({num_chunks} chunks)', 'success')
        else:
            flash(f'❌ Error: {message}', 'error')
            
    except Exception as e:
        flash(f'❌ Upload failed: {str(e)}', 'error')
    
    return redirect(url_for('admin'))


@app.route("/rebuild", methods=["POST"])
def rebuild_index():
    """Rebuild index from all uploaded PDFs."""
    
    try:
        global vectorstore, retriever
        
        with index_lock:
            print("\n🔨 Rebuilding index from scratch...")
            
            # Get all PDFs
            pdf_files = list(Path(UPLOAD_FOLDER).glob("*.pdf"))
            
            if not pdf_files:
                flash('⚠️ No PDFs to rebuild from', 'warning')
                return redirect(url_for('admin'))
            
            # Load all PDFs
            all_docs = []
            for pdf_path in pdf_files:
                try:
                    loader = PyPDFLoader(str(pdf_path))
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"⚠️ Error loading {pdf_path.name}: {e}")
            
            if not all_docs:
                flash('❌ No documents loaded', 'error')
                return redirect(url_for('admin'))
            
            # Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(all_docs)
            
            # Rebuild index
            emb = initialize_embeddings()
            vectorstore = FAISS.from_documents(chunks, emb)
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Save
            vectorstore.save_local(INDEX_PATH)
            
            flash(f'✅ Index rebuilt: {len(chunks)} chunks from {len(pdf_files)} PDFs', 'success')
            
    except Exception as e:
        flash(f'❌ Rebuild failed: {str(e)}', 'error')
    
    return redirect(url_for('admin'))


@app.route("/status")
def status():
    """API endpoint for system status."""
    stats = get_index_stats()
    
    return jsonify({
        "status": "online",
        "rag_enabled": vectorstore is not None,
        "num_vectors": stats["num_vectors"],
        "num_pdfs": stats["num_pdfs"],
        "uploaded_pdfs": stats["pdfs"]
    })


# ============================================================================
# STARTUP
# ============================================================================

def init_app():
    """Initialize app on startup."""
    print("\n" + "="*70)
    print("🏥 MEDICAL CHATBOT WITH RUNTIME VECTOR STORE")
    print("="*70)
    
    # Load or create index
    load_or_create_index()
    
    stats = get_index_stats()
    print(f"\n📊 CURRENT STATUS:")
    print(f"   Vectors in index: {stats['num_vectors']}")
    print(f"   PDFs uploaded: {stats['num_pdfs']}")
    
    if stats['num_pdfs'] > 0:
        print(f"   Files: {', '.join(stats['pdfs'][:3])}")
        if stats['num_pdfs'] > 3:
            print(f"          ... and {stats['num_pdfs'] - 3} more")
    
    print("\n🌐 ENDPOINTS:")
    print("   Chatbot:      http://localhost:5000/")
    print("   Admin Panel:  http://localhost:5000/admin")
    print("   API Status:   http://localhost:5000/status")
    print("="*70 + "\n")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    init_app()
    app.run(debug=True, port=5000, host="0.0.0.0")