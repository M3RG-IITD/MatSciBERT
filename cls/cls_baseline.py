from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

from nltk.stem import WordNetLemmatizer
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


lem = WordNetLemmatizer()


def preprocess(text):
    result = []
    text = ' '.join(text.split())
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 2:
            result.append(token)
    result = ' '.join([lem.lemmatize(w) for w in result])
    return result


splits = ['train', 'val', 'test']
dfs = {k: pd.read_csv(f'datasets/glass_non_glass/{k}.csv') for k in splits}
for split in splits:
    dfs[split]['Abstract'] = dfs[split]['Abstract'].map(preprocess)


model = make_pipeline(
    TfidfVectorizer(strip_accents='ascii', ngram_range=(1, 3), max_df=0.99, min_df=0.01),
    LogisticRegression(class_weight='balanced', C=0.3, random_state=0),
)

model.fit(dfs['train']['Abstract'], dfs['train']['Label'])

for split in ['val', 'test']:
    y_pred = model.predict(dfs[split]['Abstract'])
    accuracy = round(accuracy_score(dfs[split]['Label'], y_pred), 4)
    print(f'{split.capitalize()} accuracy: {accuracy}')
