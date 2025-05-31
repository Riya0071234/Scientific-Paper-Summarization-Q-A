print("Testing imports...")

try:
    import langchain

    print("✓ LangChain imported successfully")

    import chromadb

    print("✓ ChromaDB imported successfully")

    import sentence_transformers

    print("✓ Sentence Transformers imported successfully")

    import streamlit

    print("✓ Streamlit imported successfully")

    import fitz  # PyMuPDF

    print("✓ PyMuPDF imported successfully")

    import PyPDF2

    print("✓ PyPDF2 imported successfully")

    print("\n🎉 All packages installed successfully!")
    print("You're ready to proceed to Phase 2!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please reinstall the missing package")