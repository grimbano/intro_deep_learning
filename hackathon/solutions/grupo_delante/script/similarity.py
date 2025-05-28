import torch
import pickle
from PIL import Image
import io
import os
import requests
import base64

from transformers import ViTModel, ViTImageProcessor

import warnings
warnings.filterwarnings('ignore')

import logging
logging.disable(logging.WARNING)

from collections import defaultdict



# Change current dir to the execution place
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DB_PATH_STRUCTURE = 'embeddings/pokemon_embeddings_pkmn.pkl'





# --- Device Selection ---
# TODO: Implement device selection logic
# Hint: Check for CUDA, MPS, or fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"  # Replace with your implementation





# --- Load Pretrained Model ---
def get_model() -> ViTModel:
    """
    TODO: Implement model loading
    - Load a pretrained model (e.g., ResNet18)
    - Remove the classification head
    - Set the model to evaluation mode
    - Move the model to the appropriate device
    
    Returns:
        torch.nn.Module: The prepared model
    """

    model = ViTModel.from_pretrained('imjeffhi/pokemon_classifier').to(device)


    return model.eval()





# --- Image Preprocessing ---
# TODO: Define your image transformation pipeline
# Hint: Consider resizing, normalization, and tensor conversion
transform = ViTImageProcessor.from_pretrained(get_model().name_or_path)





class PokemonSimilarity:
    def __init__(self) -> None:
        """
        TODO: Initialize the similarity engine
        - Load the model
        - Load the database of Pokemon embeddings
        """
        self.model = get_model()
        self.db = self._load_db()


    def _load_db(self) -> dict | None:
        """
        TODO: Implement database loading
        - Look for the embeddings file in different possible locations
        - Load the pickle file containing Pokemon embeddings
        - Handle cases where the file is not found
        
        Returns:
            list: List of dictionaries containing Pokemon embeddings and labels
        """
        
        db_path = None

        try:

            if os.path.exists(DB_PATH_STRUCTURE):
                db_path = DB_PATH_STRUCTURE

            if os.path.exists(f'../{DB_PATH_STRUCTURE}'):
                db_path = f'../{DB_PATH_STRUCTURE}'
            
            with open(db_path, 'rb') as f:
                # Load the dictionary from the file
                embeddings = pickle.load(f)

            return embeddings
        
        except Exception as e:
            raise os.error(f'\nError loading embeddings database:\n{e}')



    def load_image(self, image_input) -> Image.Image:
        """
        TODO: Implement image loading
        Handle different input formats:
        - URL strings
        - Base64 encoded image strings
        - Bytes objects
        - PIL Image objects
        
        Args:
            image_input: Image in various formats
            
        Returns:
            PIL.Image: The loaded image in RGB format
        """
        if isinstance(image_input, Image.Image):
            # Already a PIL Image object
            return image_input.convert('RGB')

        elif isinstance(image_input, str): 
            # Check if it's a local file path
            if os.path.exists(image_input):
                try:
                    return Image.open(image_input).convert('RGB')
                
                except Image.UnidentifiedImageError as e:
                    raise Image.UnidentifiedImageError(f"\nCannot identify image file at path '{image_input}':\n{e}")
                
                except Exception as e:
                    raise ValueError(f"\nError loading image from local file path '{image_input}':\n{e}")

            # Check if it's a URL
            elif image_input.startswith(('http://', 'https://')):
                try:
                    response = requests.get(image_input, stream=True)
                    response.raise_for_status() # Raise an exception for bad status codes
                    return Image.open(io.BytesIO(response.content)).convert('RGB')
                
                except requests.RequestException as e:
                    raise requests.RequestException(f'\nError loading image from URL "{image_input}":\n{e}')
                
                except Exception as e:
                    raise ValueError(f'\nError processing image from URL "{image_input}":\n{e}')

            # Check if it's a Base64 encoded string
            try:
                # Base64 strings often include a prefix like "data:image/jpeg;base64,"
                # We need to remove that prefix before decoding.
                if ',' in image_input:
                    _, base64_data = image_input.split(',', 1)
                else:
                    base64_data = image_input

                decoded_image = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(decoded_image)).convert('RGB')
            
            except (base64.binascii.Error, ValueError) as e:
                # If it's not a valid Base64, it might just be an unsupported string
                # We'll let the final ValueError catch it if no other type matches.
                pass # Continue to check other types or raise final error

        elif isinstance(image_input, bytes):
            # Bytes object
            try:
                return Image.open(io.BytesIO(image_input)).convert('RGB')
            
            except Exception as e:
                raise ValueError(f'\nError loading image from bytes object:\n{e}')

        raise ValueError(f'\nUnsupported image input format: {type(image_input)}. Expected URL, Base64 string, bytes, or PIL Image.')



    def get_embedding(self, image) -> torch.Tensor:
        """
        TODO: Implement embedding generation
        Generate a feature vector for the input image using the model
        
        Args:
            image (PIL.Image): Input image to generate embedding for
            
        Returns:
            numpy.ndarray: The image embedding
        """
        
        inputs = transform(images=image, return_tensors="pt").to(device)

        last_hidden_state = self.model(**inputs).last_hidden_state

        return last_hidden_state.reshape(last_hidden_state.shape[0], -1)


    def cosine_similarity(self, a, b) -> float:
        """
        TODO: Implement cosine similarity
        Calculate the cosine similarity between two vectors
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            float: Cosine similarity score
        """
        
        return float(torch.nn.functional.cosine_similarity(a, b, dim=1))


    def find_closest_pokemon(self, image_input):
        """
        TODO: Implement Pokemon matching
        1. Load the input image
        2. Generate its embedding
        3. Compare with all Pokemon embeddings in the database
        4. Return the name of the most similar Pokemon
        
        Args:
            image_input: Image in various formats (URL, base64, bytes, PIL Image)
            
        Returns:
            str: Name of the most similar Pokemon
        """

        # Load the input_image
        image = self.load_image(image_input)

        # Generate embedding for the input image
        input_emb = self.get_embedding(image)

        # Compute similarities with all database entries
        similarities = []
        for label, emb_list in self.db.items():
            for emb in emb_list:
                similarities.append((
                    label,
                    self.cosine_similarity(input_emb, emb)
                ))

        # Sort by similarity, descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Majority voting
        data = lambda: defaultdict(float)
        summary = defaultdict(data)
        for label, similarity in similarities[:5]:
            summary[label]['votes'] += 1
            summary[label]['max_sim'] = max(summary[label]['max_sim'], similarity)

        # Sort by votes, descending. In draw case prior max_similarity
        sorted_votes = [(label, data['votes'], data['max_sim']) for label, data in summary.items()]
        sorted_votes.sort(key=lambda x: (x[1], x[2]), reverse=True)

        return sorted_votes[0][0]





if __name__ == "__main__":
    similarity_engine = PokemonSimilarity()
    print(similarity_engine.find_closest_pokemon('https://alfabetajuega.com/hero/2019/03/Squirtle-Looking-Happy.jpg?width=1200&aspect_ratio=16:9')) 
    # print(similarity_engine.find_closest_pokemon(r'C:\python\intro_deep_learning\hackathon\solutions\grupo_delante\data\testing\charmander\charmander.jpeg')) 
    