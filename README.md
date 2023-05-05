# Hyponym-Hypernym Discovery

We have the following 6 source files - 
    
    1. hearstPatterns.py - This file extracts Hearst Patterns by chunking followed by pattern matching using regular expressions.

    2. test_hearstPatterns.py - This file preprocesses the unlabeled UMBC corpus and generates hyponym-hypernym pairs.

    3. embeddings.py - This file further trains pretrained Word2Vec embeddings on hyponym-hypernym relations extracted from our dataset using negative sampling.

    4. unsupervised_approach.py - This contains an implementation of the unsupervised approach and a comparison of results from pretrained Glove embeddings and our custom trained Word2Vec embeddings.

    5. supervised_approach.py - This contains an implementation of the projection learning based supervised approach for hypernym discovery.

    6. hybrid_approach.py - This file implements a hybrid model comprising both of the supervised and unsupervised approach models and takes the best of both.


# Google Drive File Structure
- Results 
  - Embeddings 
    - hypernym-hyponym-embeddings_1A.pkl
    - hypernym-hyponym-embeddings_2A.pkl
    - hypernym-hyponym-embeddings_2B.pkl
  - hypernym discovery model
    - HH_Projection_model_1A.pt
    - HH_Projection_model_2A.pt
    - HH_Projection_model_2B.pt
  - Dictionaries 
    - hypernym-hyponym-dictionaries_1A.pkl
    - hypernym-hyponym-dictionaries_2A.pkl
    - hypernym-hyponym-dictionaries_2B.pkl
  - Scores
    - 1A_hybrid_score.txt
    - 2A_hybrid_score.txt
    - 2B_hybrid_score.txt
    - 1A_unsupervised_score.txt
    - 2A_unsupervised_score.txt
    - 2B_unsupervised_score.txt
  
  
  Google drive link - https://drive.google.com/drive/folders/1KwTvPf3Tj3PxS_Oq_V7UNoYT3YT_wjc-?usp=sharing

  Github link - https://github.com/AyushMittal1109/Hypernym-Hyponym-Discovery