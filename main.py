import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter

# 下载NLTK的必要数据（如果尚未下载）
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')  # 添加词性标记所需的数据
nltk.download('wordnet')  # 下载WordNet数据库

# 读取文本文件（Moby Dick）
with open('moby_dick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization（分词）
words = word_tokenize(text.lower())  # 将文本转换为小写并分词

# Stopwords Filtering（停用词过滤）
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

# Parts-of-Speech (POS) Tagging（词性标记）
pos_tags = nltk.pos_tag(filtered_words)

# POS Frequency（词性频率）
pos_counts = Counter(tag for word, tag in pos_tags)
common_pos = pos_counts.most_common(5)
print("Most common parts of speech and their counts:")
for tag, count in common_pos:
    print(f"{tag}: {count}")

# Lemmatization（词形还原）
lemmatizer = WordNetLemmatizer()
top_20_tokens = [word for word, _ in FreqDist(filtered_words).most_common(20)]
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in top_20_tokens]
print("\nTop 20 lemmatized tokens:")
print(lemmatized_tokens)


# 绘制条形图
plt.figure(figsize=(10, 6))
tags, counts = zip(*common_pos)
plt.bar(tags, counts)
plt.title('POS Frequency Distribution')
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
