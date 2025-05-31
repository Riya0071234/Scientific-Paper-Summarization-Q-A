import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CleanedText:
    """Represents cleaned and preprocessed text"""
    original_text: str
    cleaned_text: str
    removed_elements: Dict[str, List[str]]
    statistics: Dict[str, int]


class TextPreprocessor:
    def __init__(self):
        self.reference_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d+\)',  # (1), (2), etc.
            r'\b\w+\s+et\s+al\.\s+\(\d{4}\)',  # Author et al. (2020)
            r'\(\w+\s+\d{4}\)',  # (Author 2020)
        ]

        self.equation_patterns = [
            r'\$[^$]+\$',  # LaTeX inline equations
            r'\$\$[^$]+\$\$',  # LaTeX display equations
            r'\\begin\{equation\}.*?\\end\{equation\}',  # LaTeX equations
            r'\\begin\{align\}.*?\\end\{align\}',  # LaTeX align
        ]

        self.figure_table_patterns = [
            r'Figure\s+\d+[:.]?.*?(?=\n\n|\n[A-Z]|\n\d+\.|\nFigure|\nTable|$)',
            r'Table\s+\d+[:.]?.*?(?=\n\n|\n[A-Z]|\n\d+\.|\nFigure|\nTable|$)',
            r'Fig\.\s+\d+[:.]?.*?(?=\n\n|\n[A-Z]|\n\d+\.|\nFigure|\nTable|$)',
        ]

    def clean_scientific_text(self, text: str, preserve_structure: bool = True) -> CleanedText:
        """Clean scientific text while preserving important content"""
        original_text = text
        removed_elements = {
            'references': [],
            'equations': [],
            'figures_tables': [],
            'urls': [],
            'emails': [],
            'special_chars': []
        }

        # Track statistics
        original_length = len(text)

        # Remove or replace equations (often garbled in extraction)
        for pattern in self.equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            removed_elements['equations'].extend(matches)
            text = re.sub(pattern, '[EQUATION]', text, flags=re.DOTALL | re.IGNORECASE)

        # Handle figure and table captions (keep but mark them)
        for pattern in self.figure_table_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            removed_elements['figures_tables'].extend(matches)
            if preserve_structure:
                text = re.sub(pattern, lambda m: f'[FIGURE/TABLE: {m.group(0)[:50]}...]',
                              text, flags=re.DOTALL | re.IGNORECASE)
            else:
                text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # Clean up in-text citations (keep some, remove excessive ones)
        for pattern in self.reference_patterns:
            matches = re.findall(pattern, text)
            if len(matches) > 20:  # If too many citations, remove some
                removed_elements['references'].extend(matches)
                # Keep every 3rd citation, remove others
                citation_count = 0

                def replace_citation(match):
                    nonlocal citation_count
                    citation_count += 1
                    return match.group(0) if citation_count % 3 == 0 else ''

                text = re.sub(pattern, replace_citation, text)

        # Remove URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        removed_elements['urls'].extend(urls)
        text = re.sub(url_pattern, '', text)

        # Remove email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        removed_elements['emails'].extend(emails)
        text = re.sub(email_pattern, '', text)

        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n ', '\n', text)  # Remove spaces after newlines

        # Remove standalone numbers and short fragments
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not (line.isdigit() or len(line) < 3):
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Final cleanup
        text = text.strip()

        # Calculate statistics
        statistics = {
            'original_length': original_length,
            'cleaned_length': len(text),
            'reduction_percentage': round((1 - len(text) / original_length) * 100, 2),
            'references_removed': len(removed_elements['references']),
            'equations_replaced': len(removed_elements['equations']),
            'figures_tables_processed': len(removed_elements['figures_tables'])
        }

        return CleanedText(
            original_text=original_text,
            cleaned_text=text,
            removed_elements=removed_elements,
            statistics=statistics
        )

    def extract_key_sentences(self, text: str, min_length: int = 20) -> List[str]:
        """Extract key sentences that are likely to contain important information"""
        sentences = self.split_sentences(text)
        key_sentences = []

        # Keywords that indicate important content
        importance_keywords = [
            'conclude', 'results show', 'we found', 'demonstrates', 'significant',
            'important', 'novel', 'propose', 'method', 'approach', 'algorithm',
            'performance', 'accuracy', 'improvement', 'compared to', 'outperforms'
        ]

        for sentence in sentences:
            if len(sentence) < min_length:
                continue

            sentence_lower = sentence.lower()

            # Score sentence based on importance keywords
            importance_score = sum(1 for keyword in importance_keywords
                                   if keyword in sentence_lower)

            # Add sentences with high importance or meeting certain criteria
            if (importance_score >= 2 or
                    any(keyword in sentence_lower for keyword in ['we propose', 'our method', 'results show']) or
                    sentence.endswith('.')):
                key_sentences.append(sentence.strip())

        return key_sentences

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics"""
        # Simple sentence splitting (can be improved with nltk/spacy)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def prepare_for_chunking(self, text: str) -> str:
        """Prepare text for optimal chunking"""
        # Ensure paragraphs are clearly separated
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Add clear section breaks where detected
        section_headers = [
            'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion',
            'Abstract', 'Background', 'Related Work', 'Experiments'
        ]

        for header in section_headers:
            # Add clear breaks before section headers
            pattern = rf'\n({header})\n'
            text = re.sub(pattern, rf'\n\n--- {header.upper()} ---\n\n', text, flags=re.IGNORECASE)

        return text


# Test function
def test_text_preprocessing():
    """Test the text preprocessor"""
    preprocessor = TextPreprocessor()

    # Sample scientific text with common issues
    sample_text = """
    This is a sample paper [1] with many citations [2, 3]. 
    The equation $E = mc^2$ is important. 
    Figure 1: This shows the results of our experiment.

    We found significant improvements (p < 0.05) in our method.
    Visit http://example.com for more details.
    Contact author@university.edu for questions.

    123

    The results demonstrate that our approach outperforms 
    baseline methods by 15%.
    """

    print("Testing text preprocessing...")
    print(f"Original text length: {len(sample_text)}")
    print("\nOriginal text:")
    print(sample_text)

    # Clean the text
    cleaned = preprocessor.clean_scientific_text(sample_text)

    print(f"\nCleaned text length: {len(cleaned.cleaned_text)}")
    print(f"Reduction: {cleaned.statistics['reduction_percentage']}%")
    print("\nCleaned text:")
    print(cleaned.cleaned_text)

    print("\nRemoved elements:")
    for category, items in cleaned.removed_elements.items():
        if items:
            print(f"  {category}: {len(items)} items")
            for item in items[:2]:  # Show first 2 items
                print(f"    - {item[:50]}...")

    # Test key sentence extraction
    key_sentences = preprocessor.extract_key_sentences(cleaned.cleaned_text)
    print(f"\nKey sentences extracted: {len(key_sentences)}")
    for sentence in key_sentences:
        print(f"  - {sentence}")

    print("\n✓ Text preprocessing test completed!")


if __name__ == "__main__":
    test_text_preprocessing()