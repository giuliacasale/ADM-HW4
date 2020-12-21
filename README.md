# ADM-HOMEWORK 4: HARD CODING
The aim of this homework was to write important algorithms and functions from scratch.
The homework was divided into three main parts:
1. **HASHING**: we had to write our own hash function and then use it to implment the hyperloglog algorithm. The final goal was to return the *cardinality of the dataset* and the *error of the filter*.
2. **CLUSTERING**: starting from [this](https://www.kaggle.com/snap/amazon-fine-food-reviews) Kaggle dataset of Amazon's reviews and
3. **ALGORITHMIC QUESTION**: where we had to prove the running time of a sorting algorithm

The repository is organized as follows:
- `point_1_3.ipynb` that contains all the functions and the code to fully answer to the hashing and algorithmic question
- `point_2.ipynb` that contains all the steps we followed to pre-process the text, apply the SVD Method, implementing the clustering, and all the visualization of the results with our comments
- `functions.py` that contains the majority of the functions used in the notebook that contains the cluster question, all described for what they do.

-----
**N.B.** To prepare the data to give in input to the K-Means Algorithm, we did a lot of preliminary operations that took a long time to process. We descibed in the notebook step by step each choice we made, but for an easier work, a lot of these functions were runned only the first time and then saved into a file (pkl or csv) and then we load back directly the final output every time we restarted the notebook. These operations can be recognized in the notebook since we then converted them into a markdown cell. We will leave some links below so that those files  can be  directly download and just read inside the notebook, without re-creating them

**Links for the download of the following files:**
- clean_dataset.csv: https://www.mediafire.com/file/c0b3d3hjlivr9k2/clean_dataset.csv/file
- dictionary.pkl: https://www.mediafire.com/file/ilyhl9bxfaxvr9b/dictionary.pickle/file
- final_dataset.csv: https://www.mediafire.com/file/nfpmqjuyy6c3iy6/final_dataset.csv/file
- frequencies.pkl: https://www.mediafire.com/file/mx1npybd355ylsh/frequencies.pickle/file
- new_data.pkl: https://www.mediafire.com/file/z3d3prd20p9oq36/new_data.pickle/file
- tf_idf_scores.pkl: https://www.mediafire.com/file/87ddoy3z6xgbrxz/tf_idf_scores.pickle/file
- tf_idf_scores_dataset.csv: https://www.mediafire.com/file/qpy3979eewwmghl/tf_idf_scores_dataframe.csv/file
