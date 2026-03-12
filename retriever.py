import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class NCORetriever:
    def __init__(self, 
                 data_path: str = "data/nco_2015.pkl",
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_embeddings: bool = True):
        
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        
        # Load data from pickle file
        self.data, self.precomputed_embeddings = self._load_data()
        
        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Use precomputed embeddings if available, otherwise compute
        if self.precomputed_embeddings is not None:
            self.embeddings = self.precomputed_embeddings
            logger.info(f"Using precomputed embeddings with shape {self.embeddings.shape}")
        else:
            self.embeddings = self._load_embeddings()
        
        logger.info(f"Retriever initialized with {len(self.data)} occupations")
    
    def _load_data(self):
        """Load NCO 2015 data from pickle file"""
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return self._create_sample_data(), None
        
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded data type: {type(data)}")
            
            # Handle different data types
            if isinstance(data, pd.DataFrame):
                # If it's already a DataFrame
                df = data
                embeddings = None
                
            elif isinstance(data, np.ndarray):
                # If it's just embeddings, create sample data
                logger.warning("Pickle file contains embeddings array. Creating sample data.")
                df = self._create_sample_data()
                embeddings = data
                
            elif isinstance(data, dict):
                # If it's a dictionary, try to extract data and embeddings
                df = None
                embeddings = None
                
                # Look for DataFrame in dict
                for key in ['data', 'df', 'dataframe', 'occupations']:
                    if key in data and isinstance(data[key], pd.DataFrame):
                        df = data[key]
                        break
                
                # Look for embeddings in dict
                for key in ['embeddings', 'vectors', 'features']:
                    if key in data and isinstance(data[key], np.ndarray):
                        embeddings = data[key]
                        break
                
                # If no DataFrame found, create one from array data
                if df is None:
                    # Try to create DataFrame from the data
                    for key, value in data.items():
                        if isinstance(value, (list, np.ndarray)) and not isinstance(value, np.ndarray):
                            df = pd.DataFrame(value)
                            break
                
                if df is None:
                    df = self._create_sample_data()
                    
            elif isinstance(data, list):
                # If it's a list, try to convert to DataFrame
                try:
                    df = pd.DataFrame(data)
                except:
                    df = self._create_sample_data()
                embeddings = None
                
            else:
                logger.error(f"Unexpected data type: {type(data)}")
                return self._create_sample_data(), None
            
            # Ensure required columns exist
            required = ['title', 'nco_2015', 'description']
            for col in required:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found. Adding empty column.")
                    df[col] = ""
            
            logger.info(f"Loaded {len(df)} occupations")
            return df, embeddings
            
        except Exception as e:
            logger.error(f"Error loading pickle data: {e}")
            return self._create_sample_data(), None
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        logger.warning("Creating sample NCO data for testing")
        sample_data = {
            'title': [
                'Mechanical Engineer',
                'Software Developer',
                'Nurse',
                'Electrician',
                'Teacher'
            ],
            'nco_2015': [
                '2144',
                '2512',
                '2221',
                '7411',
                '2310'
            ],
            'description': [
                'Design and develop mechanical systems, work with machinery, create technical drawings, and solve engineering problems.',
                'Design, develop, and maintain software applications, work with programming languages, and solve technical problems.',
                'Provide patient care, administer medications, monitor health conditions, and work in healthcare settings.',
                'Install and maintain electrical systems, read blueprints, troubleshoot electrical problems, and ensure safety compliance.',
                'Plan and deliver lessons, assess student progress, create learning materials, and work in educational settings.'
            ]
        }
        return pd.DataFrame(sample_data)
    
    def _load_embeddings(self) -> np.ndarray:
        """Load or compute embeddings"""
        cache_file = self.data_path.with_suffix('.embeddings.pkl')
        
        if self.cache_embeddings and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings from cache: {cache_file}")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
        
        # Compute embeddings
        logger.info("Computing document embeddings...")
        texts = self.data['title'] + " " + self.data['description']
        embeddings = self.model.encode(texts.tolist(), show_progress_bar=True)
        
        # Save to cache
        if self.cache_embeddings:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embeddings, f)
                logger.info(f"Saved embeddings to cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save embeddings cache: {e}")
        
        return embeddings
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar occupations with normalized scores"""
        if not query.strip():
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])[0]
            
            # Normalize embeddings for cosine similarity
            if not hasattr(self, 'embeddings_normalized'):
                # Normalize document embeddings (do this once)
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                # Avoid division by zero
                norms = np.where(norms == 0, 1, norms)
                self.embeddings_normalized = self.embeddings / norms
            
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding_normalized = query_embedding / query_norm
            else:
                query_embedding_normalized = query_embedding
            
            # Compute cosine similarity (values between -1 and 1)
            similarities = np.dot(self.embeddings_normalized, query_embedding_normalized)
            
            # Convert to percentage (0-100%) for better display
            # Map from [-1,1] to [0,1] for scores below 0
            similarities_percent = (similarities + 1) / 2  # Maps -1->0, 0->0.5, 1->1
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Prepare results
            results = []
            for idx in top_indices:
                # Ensure score is a float, never None
                raw_score = float(similarities[idx])
                percent_score = float(similarities_percent[idx])
                
                results.append({
                    'title': str(self.data.iloc[idx]['title']),
                    'nco_2015': str(self.data.iloc[idx]['nco_2015']),
                    'description': str(self.data.iloc[idx]['description']),
                    'similarity_score': percent_score,  # Always a float between 0-1
                    'raw_score': raw_score  # Always a float between -1 and 1
                })
            
            # Sort by similarity_score descending
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            # Return empty list with error handling
            return []