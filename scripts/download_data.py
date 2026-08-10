import os
import soundfile as sf
from datasets import load_dataset

def main():
    print("Loading GTZAN dataset from huggingface...")
    dataset = load_dataset("marsyas/gtzan", split="train")
    
    base_dir = "data/raw/genres_original"
    os.makedirs(base_dir, exist_ok=True)
    
    # GTZAN has 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for g in genres:
        os.makedirs(os.path.join(base_dir, g), exist_ok=True)
        
    counts = {g: 0 for g in genres}
    
    for i, item in enumerate(dataset):
        genre_id = item['genre']
        genre_name = genres[genre_id]
        
        audio = item['audio']
        sample_rate = audio['sampling_rate']
        array = audio['array']
        
        filename = f"{genre_name}.{counts[genre_name]:05d}.wav"
        filepath = os.path.join(base_dir, genre_name, filename)
        
        sf.write(filepath, array, sample_rate)
        counts[genre_name] += 1
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} files")
            
    print("Done downloading raw audio.")

if __name__ == "__main__":
    main()
