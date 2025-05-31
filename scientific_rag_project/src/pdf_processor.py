import fitz  # PyMuPDF
import PyPDF2
import re
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentSection:
    """Represents a section of a scientific paper"""
    title: str
    content: str
    section_type: str  # 'title', 'abstract', 'introduction', 'methods', etc.
    page_numbers: List[int]


class PDFProcessor:
    def __init__(self):
        self.common_section_headers = [
            'abstract', 'introduction', 'methods', 'methodology', 'results',
            'discussion', 'conclusion', 'references', 'acknowledgments',
            'background', 'related work', 'experiments', 'evaluation',
            'future work', 'limitations'
        ]

    def extract_text_pymupdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text using PyMuPDF (better for complex layouts)"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            metadata = {
                'total_pages': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'pages_text': []
            }

            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()

                # Clean page text
                page_text = self.clean_page_text(page_text)
                full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
                metadata['pages_text'].append({
                    'page_num': page_num + 1,
                    'text': page_text
                })

            doc.close()
            return full_text, metadata

        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            return "", {}

    def extract_text_pypdf2(self, pdf_path: str) -> Tuple[str, Dict]:
        """Fallback extraction using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                metadata = {
                    'total_pages': len(pdf_reader.pages),
                    'title': pdf_reader.metadata.get('/Title', '') if pdf_reader.metadata else '',
                    'author': pdf_reader.metadata.get('/Author', '') if pdf_reader.metadata else '',
                    'pages_text': []
                }

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    page_text = self.clean_page_text(page_text)
                    full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
                    metadata['pages_text'].append({
                        'page_num': page_num + 1,
                        'text': page_text
                    })

                return full_text, metadata

        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return "", {}

    def clean_page_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        # Remove common headers/footers
        text = re.sub(r'^(doi:|DOI:).*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace("’", "'").replace('“', '"').replace('”', '"')


        # Remove URLs (often broken across lines)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        return text.strip()

    def extract_paper_structure(self, text: str) -> List[DocumentSection]:
        """Identify and extract paper sections"""
        sections = []

        # Try to identify title (usually first substantial text)
        lines = text.split('\n')
        title_candidates = []

        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            line = line.strip()
            if len(line) > 10 and not line.lower().startswith('page'):
                title_candidates.append((i, line))

        # Extract title (first meaningful line)
        title = title_candidates[0][1] if title_candidates else "Unknown Title"
        sections.append(DocumentSection(
            title="Title",
            content=title,
            section_type="title",
            page_numbers=[1]
        ))

        # Split text by potential section headers
        current_section = "introduction"
        current_content = []
        current_pages = [1]

        for line in lines:
            line = line.strip()

            # Check if line is a section header
            if self.is_section_header(line):
                # Save previous section
                if current_content:
                    sections.append(DocumentSection(
                        title=current_section.title(),
                        content='\n'.join(current_content),
                        section_type=current_section.lower(),
                        page_numbers=current_pages
                    ))

                # Start new section
                current_section = self.normalize_section_name(line)
                current_content = []
                current_pages = self.extract_page_numbers(line)

            else:
                if line and not line.startswith('---'):  # Skip page markers
                    current_content.append(line)
                    # Update page numbers if we find page markers
                    if line.startswith('--- Page'):
                        page_num = self.extract_page_number(line)
                        if page_num and page_num not in current_pages:
                            current_pages.append(page_num)

        # Add final section
        if current_content:
            sections.append(DocumentSection(
                title=current_section.title(),
                content='\n'.join(current_content),
                section_type=current_section.lower(),
                page_numbers=current_pages
            ))

        return sections

    def is_section_header(self, line: str) -> bool:
        """Check if a line is likely a section header"""
        line_lower = line.lower().strip()

        # Check against common section headers
        for header in self.common_section_headers:
            if header in line_lower:
                # Additional checks to avoid false positives
                if len(line) < 100 and not line.endswith('.'):
                    return True

        # Check for numbered sections (1. Introduction, 2.1 Methods, etc.)
        if re.match(r'^\d+\.?\s+[A-Z][a-z]', line):
            return True

        return False

    def normalize_section_name(self, header: str) -> str:
        """Normalize section header to standard name"""
        header_lower = header.lower().strip()

        for standard_name in self.common_section_headers:
            if standard_name in header_lower:
                return standard_name

        # Remove numbers and cleanup
        cleaned = re.sub(r'^\d+\.?\s*', '', header).strip()
        return cleaned.lower() if cleaned else "unknown"

    def extract_page_numbers(self, text: str) -> List[int]:
        """Extract page numbers from text"""
        page_numbers = re.findall(r'--- Page (\d+) ---', text)
        return [int(p) for p in page_numbers]

    def extract_page_number(self, line: str) -> int:
        """Extract single page number from page marker"""
        match = re.search(r'--- Page (\d+) ---', line)
        return int(match.group(1)) if match else None

    def process_pdf(self, pdf_path: str) -> Dict:
        """Main method to process a PDF file"""
        print(f"Processing: {pdf_path}")

        # Try PyMuPDF first, fallback to PyPDF2
        full_text, metadata = self.extract_text_pymupdf(pdf_path)

        if not full_text:
            print("PyMuPDF failed, trying PyPDF2...")
            full_text, metadata = self.extract_text_pypdf2(pdf_path)

        if not full_text:
            raise Exception("Both PDF extraction methods failed")

        # Extract paper structure
        sections = self.extract_paper_structure(full_text)

        # Compile results
        result = {
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path),
            'full_text': full_text,
            'metadata': metadata,
            'sections': sections,
            'total_characters': len(full_text),
            'processing_success': True
        }

        print(f"✓ Successfully processed {metadata['total_pages']} pages")
        print(f"✓ Extracted {len(sections)} sections")
        print(f"✓ Total characters: {len(full_text):,}")

        return result

# Test function
def test_pdf_processing():
    """Test the PDF processor with sample files"""
    processor = PDFProcessor()
    pdf_dir = Path("data/pdfs")

    if not pdf_dir.exists():
        print("❌ PDF directory not found. Please add PDFs to data/pdfs/")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in data/pdfs/")
        return

    print(f"Found {len(pdf_files)} PDF files to process...")

    for pdf_file in pdf_files[:2]:  # Process first 2 files for testing
        try:
            result = processor.process_pdf(str(pdf_file))

            print(f"\n📄 File: {result['file_name']}")
            print(f"📊 Pages: {result['metadata']['total_pages']}")
            print(f"📝 Sections found:")

            for section in result['sections']:
                content_preview = section.content[:100] + "..." if len(section.content) > 100 else section.content
                print(f"  - {section.title}: {len(section.content)} chars")
                print(f"    Preview: {content_preview}")

            print("-" * 50)

        except Exception as e:
            print(f"❌ Failed to process {pdf_file}: {e}")

if __name__ == "__main__":
    test_pdf_processing()