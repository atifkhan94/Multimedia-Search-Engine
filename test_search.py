import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from src.search.search_engine import SearchEngine

def test_search_engine():
    try:
        # Initialize the search engine
        print("Initializing search engine...")
        engine = SearchEngine()
        print("Search engine initialized successfully!")

        # Test search functionality
        print("\nTesting search functionality...")
        query = "test query"
        result = engine.search(query, media_type='all')
        print(f"\nSearch Results for '{query}':\n{result}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    test_search_engine()