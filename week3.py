# %%
import numpy as np
from scipy.spatial import distance

pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

#Euclidean distance
euclidean_dist = distance.euclidean(pointA, pointB)
print("Euclidean Distance:", euclidean_dist)

#Similarity (inverse of distance)
similarity_euclidean = 1 / (1+euclidean_dist)
print("Euclidean Similarity:", similarity_euclidean)

# %%
#manhatten distance
manhattan_dist = distance.cityblock(pointA, pointB)
print("Manhattan Distance:", manhattan_dist)

#similarity (inverse of distance)
similarity_manhattan = 1 / (1 + manhattan_dist)
print("Manhattan Distance:", similarity_manhattan)
# %%
#Minkowski distance with p=3
minkowski_dist_p3 = distance.minkowski(pointA, pointB, p=3)
print("Minkowski Distance (p=3): ", minkowski_dist_p3)

#Similarity (inverse of distance)
similarity_minkowski = 1 / (1 + minkowski_dist_p3)
print("Minkowski Similarity (p=3): ", similarity_minkowski)

# %%
import pandas as pd

df = pd.DataFrame({
    'X': [10, 20, 30, 40, 50],
    'Y': [12, 24, 33, 45, 60]  })

corr_matrix = df.corr(method='pearson')
print("Pearson Correlation Matrix:\n", corr_matrix)
# %%
import pandas as pd

df = pd.DataFrame({
    'X': [10, 20, 30, 40, 50],
    'Y': [10, 20, 30, 40, 50]  })

corr_matrix = df.corr(method='pearson')
print("Pearson Correlation Matrix:\n", corr_matrix)

# %%
import pandas as pd

df = pd.read_csv("Iris.csv")
print(df.corr(method='pearson', numeric_only=float))
# %%

import pandas as pd
from scipy.stats import spearmanr

df = pd.DataFrame({
    'x': [10, 20, 30, 40, 50],
    'Y': [12, 18, 33, 47, 55]
})

corr_value, p_value = spearmanr(df['x'], df['Y'])
print(f"Spearman Correlation Coefficient: {corr_value}")
print(f"P-value: {p_value}")

# %%
import pandas as pd

df = pd.read_csv("Iris.csv")
print(df.corr(method='spearman', numeric_only=float))
# %%
#create a dataset with 5 student records for 5 subjects and calculate the correlation between the subjects using Pearson and Spearman methods.
import pandas as pd
from scipy.stats import spearmanr

df = pd.DataFrame({
    'names': ['A', 'B', 'C', 'D', 'E'],
    'AIML': [67, 55, 30, 65, 94],
    'DS': [78, 45, 67, 89, 90],
    'DBMS': [56, 78, 90, 45, 67],
    'OS': [45, 67, 89, 90, 78],
    'CN': [90, 78, 56, 67, 45]
})

pearson_corr = df.corr(method='pearson', numeric_only=float)
print("Pearson Correlation Matrix:\n", pearson_corr)

spearman_corr=df.corr(method='spearman', numeric_only=float)
print("Spearman Correlation Matrix:\n", spearman_corr)
# %%

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")

    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

s1 = "I am krishita"
s2 = "I study in KL"

dist = hamming_distance(s1, s2)
print(f"Hamming Distance between '{s1}' and '{s2}' :{dist}")

# %%
def jaccard_index(str1, str2):
    set1, set2 = set(str1.split()), set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection)/len(union)

s1 = "data science is fun"
s2 = "science makes data useful"

print("Jaccard Index:", jaccard_index(s1, s2))

# %%

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

s1= "data science is fun"
s2 = "science makes data useful" 
vectorizer = CountVectorizer().fit([s1,s2])
vectors = vectorizer.transform([s1, s2])
cos_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
print("Cosine Similarity:", cos_sim)
# %%
def lcs_length(X,Y):
    m, n = len(X), len(Y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j]+1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]
 
seq1="ABCDEF"
seq2="AEBDF"
length = lcs_length(seq1,seq2)
print(f"Longest common subsequence length between '{seq1}' and '{seq2}': {length}")
# %%
