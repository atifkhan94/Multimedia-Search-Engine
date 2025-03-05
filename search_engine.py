import numpy as np
from typing import List, Dict, Union
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import spacy
from ..processor.multimedia_processor import MultimediaProcessor

class SearchEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.multimedia_processor = MultimediaProcessor()
        self._setup_nlp()

    def _setup_nlp(self):
        """Initialize NLP components."""
        try:
            # Download required NLTK data
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            
            # Initialize spaCy
            self.nlp = spacy.load('en_core_web_sm')
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP components: {e}")
            raise

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for search."""
        try:
            # Tokenize and convert to lowercase
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and token.isalnum()
            ]
            
            return tokens
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return []

    def search(self, query: str, media_type: str = 'all') -> List[Dict[str, any]]:
        """Search for multimedia content based on text query.

        Args:
            query (str): Search query
            media_type (str): Type of media to search for ('image', 'video', 'all')

        Returns:
            List of search results with relevance scores
        """
        try:
            # Preprocess query
            processed_query = self.preprocess_text(query)
            
            # Perform semantic analysis using spaCy
            doc = self.nlp(query)
            
            # Extract named entities and key phrases
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Get media files based on media type
            media_files = self._get_media_files(media_type)
            results = []

            for media_file in media_files:
                # Process media file based on type
                if media_file.endswith(('.jpg', '.jpeg', '.png')):
                    if media_type in ['image', 'all']:
                        processed_media = self.multimedia_processor.process_image(media_file)
                elif media_file.endswith(('.mp4', '.avi', '.mov')):
                    if media_type in ['video', 'all']:
                        processed_media = self.multimedia_processor.process_video(media_file)
                else:
                    continue

                if not processed_media:
                    continue

                # Calculate text similarity
                text_similarity = self._calculate_text_similarity(processed_query, processed_media)
                
                # Calculate feature similarity
                feature_similarity = 0.0
                if 'features' in processed_media:
                    query_features = self._extract_query_features(query)
                    feature_similarity = self.calculate_similarity(
                        query_features,
                        np.array(processed_media['features'])
                    )

                # Combine similarities with weights
                total_similarity = 0.7 * text_similarity + 0.3 * feature_similarity

                results.append({
                    'path': processed_media['path'],
                    'similarity': total_similarity,
                    'text_similarity': text_similarity,
                    'feature_similarity': feature_similarity,
                    'classifications': processed_media.get('classifications', []),
                    'media_type': 'video' if media_file.endswith(('.mp4', '.avi', '.mov')) else 'image'
                })

            # Rank results by similarity
            ranked_results = self.rank_results(results)

            return [{
                'query': query,
                'processed_tokens': processed_query,
                'entities': entities,
                'results': ranked_results
            }]
            
        except Exception as e:
            self.logger.error(f"Error performing search: {e}")
            return []

    def _get_media_files(self, media_type: str) -> List[str]:
        """Get list of media files based on type."""
        media_files = [
            'media/european-shorthair-8601492_640.jpg',
            'media/pexels-pixabay-45201.jpg',
            'media/5280134-uhd_4096_2160_30fps.mp4'
        ]
        return media_files

    def _calculate_text_similarity(self, query_tokens: List[str], media_data: Dict) -> float:
        """Calculate text similarity between query and media metadata."""
        try:
            # Extract text from media classifications
            media_text = ' '.join([c['label'] for c in media_data.get('classifications', [])])
            media_tokens = self.preprocess_text(media_text)
            
            # Calculate Jaccard similarity
            query_set = set(query_tokens)
            media_set = set(media_tokens)
            
            if not query_set or not media_set:
                return 0.0
                
            intersection = len(query_set.intersection(media_set))
            union = len(query_set.union(media_set))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating text similarity: {e}")
            return 0.0

    def _extract_query_features(self, query: str) -> np.ndarray:
        """Extract feature vector from query text."""
        # Use spaCy document vector as query features
        doc = self.nlp(query)
        return doc.vector

    def calculate_similarity(self, query_features: np.ndarray, target_features: np.ndarray) -> float:
        """Calculate similarity between query and target features."""
        try:
            # Using cosine similarity
            similarity = np.dot(query_features, target_features) / \
                       (np.linalg.norm(query_features) * np.linalg.norm(target_features))
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def rank_results(self, results: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Rank search results based on relevance scores."""
        try:
            return sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)
        except Exception as e:
            self.logger.error(f"Error ranking results: {e}")
            return results