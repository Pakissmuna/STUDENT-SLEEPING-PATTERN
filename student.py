# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
df = pd.read_csv(r'student_sleep_patterns.csv')
df

# Drop unnecessary columns
df = df.drop(['Student_ID', 'Weekday_Sleep_Start', 'Weekend_Sleep_Start',
              'Weekday_Sleep_End', 'Weekend_Sleep_End'], axis=1)

# Remove suffixes
df['University_Year'] = df['University_Year'].str.replace('st Year', '')
df['University_Year'] = df['University_Year'].str.replace('nd Year', '')
df['University_Year'] = df['University_Year'].str.replace('rd Year', '')
df['University_Year'] = df['University_Year'].str.replace('th Year', '')

df['University_Year'] = df['University_Year'].astype(int)

# Visualize Sleep Quality by different columns
columns = ['Age', 'Gender', 'University_Year', 'Screen_Time', 'Caffeine_Intake']
for col in columns:
    x = df.groupby([col])['Sleep_Quality'].mean().reset_index()
    sns.lineplot(x=col, y='Sleep_Quality', data=x)
    plt.title(f'Sleep Quality by {col}')
    plt.show()

# Countplot for caffeine intake by gender
sns.countplot(data=df, x='Caffeine_Intake', hue='Gender')
plt.title('Which gender takes more caffeine?')
plt.show()

# One-hot encoding and normalization
X = pd.get_dummies(df)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
normalized_X = normalize(scaled_X)

normalized_X = pd.DataFrame(normalized_X)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(normalized_X)

X_pca = pd.DataFrame(X_pca)
X_pca.columns = ['P1', 'P2']