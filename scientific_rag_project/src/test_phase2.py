from pdf_processor import PDFProcessor
from text_preprocessor import TextPreprocessor
import json
from pathlib import Path


def test_complete_pipeline():
    """Test the complete PDF processing pipeline"""
    print("=== Testing Complete PDF Processing Pipeline ===\n")

    # Initialize processors
    pdf_processor = PDFProcessor()
    text_processor = TextPreprocessor()

    # Check for PDF files
    pdf_dir = Path("data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in data/pdfs/")
        print("Please add at least one scientific paper PDF to continue.")
        return False

    print(f"Found {len(pdf_files)} PDF files to process\n")

    results = []

    for pdf_file in pdf_files[:2]:  # Process first 2 files
        try:
            print(f"📄 Processing: {pdf_file.name}")
            print("-" * 40)

            # Step 1: Extract text from PDF
            pdf_result = pdf_processor.process_pdf(str(pdf_file))

            print(f"✓ Extracted {pdf_result['metadata']['total_pages']} pages")
            print(f"✓ Found {len(pdf_result['sections'])} sections")

            # Step 2: Clean and preprocess text
            cleaned_result = text_processor.clean_scientific_text(
                pdf_result['full_text'],
                preserve_structure=True
            )

            print(f"✓ Text cleaned: {cleaned_result.statistics['reduction_percentage']}% reduction")
            print(f"✓ Original: {cleaned_result.statistics['original_length']:,} chars")
            print(f"✓ Cleaned: {cleaned_result.statistics['cleaned_length']:,} chars")

            # Step 3: Extract key sentences
            key_sentences = text_processor.extract_key_sentences(cleaned_result.cleaned_text)
            print(f"✓ Extracted {len(key_sentences)} key sentences")

            # Step 4: Prepare for chunking
            chunk_ready_text = text_processor.prepare_for_chunking(cleaned_result.cleaned_text)

            # Compile results
            complete_result = {
                'file_info': {
                    'name': pdf_file.name,
                    'path': str(pdf_file)
                },
                'pdf_extraction': pdf_result,
                'text_cleaning': {
                    'cleaned_text': cleaned_result.cleaned_text,
                    'statistics': cleaned_result.statistics,
                    'removed_elements_count': {k: len(v) for k, v in cleaned_result.removed_elements.items()}
                },
                'key_sentences': key_sentences,
                'chunk_ready_text': chunk_ready_text
            }

            results.append(complete_result)

            # Show sample sections
            print("\n📋 Sections found:")
            for section in pdf_result['sections'][:5]:  # Show first 5 sections
                content_preview = section.content[:100] + "..." if len(section.content) > 100 else section.content
                print(f"  {section.title}: {len(section.content)} chars")
                print(f"    {content_preview}\n")

            # Show sample key sentences
            print("🔑 Sample key sentences:")
            for sentence in key_sentences[:3]:  # Show first 3
                print(f"  - {sentence}\n")

            print("✅ Processing completed successfully!\n")
            print("=" * 50)

        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            continue

    # Save results for next phase
    if results:
        output_file = Path("data/processed/phase2_results.json")
        output_file.parent.mkdir(exist_ok=True)

        # Convert results to JSON-serializable format
        json_results = []
        for result in results:
            json_result = {
                'file_info': result['file_info'],
                'sections': [
                    {
                        'title': section.title,
                        'content': section.content,
                        'section_type': section.section_type,
                        'page_numbers': section.page_numbers
                    }
                    for section in result['pdf_extraction']['sections']
                ],
                'metadata': result['pdf_extraction']['metadata'],
                'cleaned_text': result['text_cleaning']['cleaned_text'],
                'statistics': result['text_cleaning']['statistics'],
                'key_sentences': result['key_sentences']
            }
            json_results.append(json_result)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {output_file}")
        print(f"📊 Successfully processed {len(results)} files")

        return True

    return False


def show_processing_summary():
    """Show summary of processed files"""
    results_file = Path("data/processed/phase2_results.json")

    if not results_file.exists():
        print("No processing results found. Run test_complete_pipeline() first.")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("=== Processing Summary ===")
    print(f"Total files processed: {len(results)}")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['file_info']['name']}")
        print(f"   Pages: {result['metadata']['total_pages']}")
        print(f"   Sections: {len(result['sections'])}")
        print(f"   Original text: {result['statistics']['original_length']:,} chars")
        print(f"   Cleaned text: {result['statistics']['cleaned_length']:,} chars")
        print(f"   Key sentences: {len(result['key_sentences'])}")

        section_types = [s['section_type'] for s in result['sections']]
        unique_sections = list(set(section_types))
        print(f"   Section types: {', '.join(unique_sections)}")


if __name__ == "__main__":
    # Run complete pipeline test
    success = test_complete_pipeline()

    if success:
        print("\n" + "=" * 60)
        show_processing_summary()
        print("\n🎉 Phase 2 completed successfully!")
        print("Ready for Phase 3: Text Chunking & Embedding Generation")
    else:
        print("\n❌ Phase 2 testing failed. Please fix issues before proceeding.")