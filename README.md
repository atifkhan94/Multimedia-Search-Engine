# Multimedia Search Engine

A powerful search engine that combines computer vision, natural language processing, and deep learning to enable intelligent search across multimedia content including images and videos.

## Features

- **Multimedia Processing**
  - Image classification using TensorFlow MobileNetV2
  - Video frame analysis with OpenCV
  - Feature extraction using OpenVINO

- **Intelligent Search**
  - Natural language query processing
  - Text similarity matching using Jaccard similarity
  - Feature-based similarity using cosine similarity
  - Named entity recognition with spaCy

- **Supported Media Types**
  - Images (jpg, jpeg, png)
  - Videos (mp4, avi, mov)

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK data and spaCy model:
   ```python
   python -m spacy download en_core_web_sm
   ```

## Usage

```python
from src.search.search_engine import SearchEngine

# Initialize the search engine
engine = SearchEngine()

# Perform a search
results = engine.search("cat picture", media_type='all')
```

## Project Structure

```
├── media/                  # Sample media files
├── src/
│   ├── crawler/           # Web crawler implementation
│   ├── processor/         # Multimedia processing modules
│   └── search/            # Core search engine implementation
└── test_search.py        # Test script
```

## Technical Details

### Search Algorithm

The search engine combines multiple similarity metrics:
- Text similarity (70% weight)
- Feature similarity (30% weight)

### Media Processing

- **Images**: Processed using TensorFlow MobileNetV2 for classification and OpenVINO for feature extraction
- **Videos**: Analyzed frame by frame with configurable sampling rate

### Dependencies

- TensorFlow
- OpenVINO
- spaCy
- NLTK
- OpenCV
- NumPy
- PIL

## License

MIT License