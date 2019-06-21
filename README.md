# Naive-Bayes

Naive-Bayes classifier is a **Probabilistic Generative Model**.

Naive-Bayes classifier is a **Supervised** machine learning algorithm.
 
 Two variants of the Naive Bayes algorithm with smoothing as applied to text categorization.

 Why is it "Naive"?

   1. A document `d` is simply represented as a bag of words, i.e., features.
   
   2. Position of the words doesn't matter; given class, the features are independent.
   
    
 *Goal*: Predcit the class `c` of document `d` given its features.

 *How?* Pick the most likely class, i.e., the __maximum a posteriori probability__ estimate of the document class:

<a href="https://www.codecogs.com/eqnedit.php?latex=c_{\textrm{MAP}}=\arg&space;\max_{c\in&space;C}&space;\mathbb{P}[c|d]=\arg&space;\max_{c\in&space;C}&space;\frac{\mathbb{P}[d|c]\mathbb{P}[c]}{\mathbb{P}[d]}=\arg&space;\max_{c\in&space;C}&space;\mathbb{P}[d|c]\mathbb{P}[c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{\textrm{MAP}}=\arg&space;\max_{c\in&space;C}&space;\mathbb{P}[c|d]=\arg&space;\max_{c\in&space;C}&space;\frac{\mathbb{P}[d|c]\mathbb{P}[c]}{\mathbb{P}[d]}=\arg&space;\max_{c\in&space;C}&space;\mathbb{P}[d|c]\mathbb{P}[c]" title="c_{\textrm{MAP}}=\arg \max_{c\in C} \mathbb{P}[c|d]=\arg \max_{c\in C} \frac{\mathbb{P}[d|c]\mathbb{P}[c]}{\mathbb{P}[d]}=\arg \max_{c\in C} \mathbb{P}[d|c]\mathbb{P}[c]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=c_{\textrm{NB}}=c_{\textrm{MAP}}=\arg&space;\max_{c\in&space;C}&space;\mathbb{P}[c]\Pi_{i=1}^{\&hash;\textrm{Features}}\mathbb{P}[w_i|c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{\textrm{NB}}=c_{\textrm{MAP}}=\arg&space;\max_{c\in&space;C}&space;\mathbb{P}[c]\Pi_{i=1}^{\&hash;\textrm{Features}}\mathbb{P}[w_i|c]" title="c_{\textrm{NB}}=c_{\textrm{MAP}}=\arg \max_{c\in C} \mathbb{P}[c]\Pi_{i=1}^{\#\textrm{Features}}\mathbb{P}[w_i|c]" /></a>

* To avoid underflow, we use the sum of the log of probabilities instead of product.

* If a word in a test document does not appear in the training set at all, i.e., none of the classes, skip that word.

Since <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{P}[c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{P}[c]" title="\mathbb{P}[c]" /></a> is unknown, we use maximum likelihood estimation: <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{\mathbb{P}}[c]=\frac{\&hash;\textrm{documents&space;of&space;class&space;}&space;c}{\text{Total&space;\&hash;&space;of&space;documents}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{\mathbb{P}}[c]=\frac{\&hash;\textrm{documents&space;of&space;class&space;}&space;c}{\text{Total&space;\&hash;&space;of&space;documents}}" title="\widehat{\mathbb{P}}[c]=\frac{\#\textrm{ documents of class } c}{\text{Total \# of documents}}" /></a>

The model has <a href="https://www.codecogs.com/eqnedit.php?latex=O(\&hash;\textrm{Features}\times\&hash;\textrm{Classes}&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O(\&hash;\textrm{Features}\times\&hash;\textrm{Classes}&space;)" title="O(\#\textrm{Features}\times\#\textrm{Classes} )" /></a> number of parameters.

In this project two variants of Naive Bayes are implemented: 

  __Type1__ 
  
   In this case, a document `d` has as many features as the number of words in it and a "token feature" has as many possible values as words in the vocabulary (all words in training files).
   
<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{\mathbb{P}}[w_i|c]=\frac{\textrm{count}(w_i,c)}{\underbrace{\sum_{\textrm{All&space;words&space;}w}\textrm{count}(w,c)}_{\textrm{count}(c)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{\mathbb{P}}[w_i|c]=\frac{\textrm{count}(w_i,c)}{\underbrace{\sum_{\textrm{All&space;words&space;}w}\textrm{count}(w,c)}_{\textrm{count}(c)}}" title="\widehat{\mathbb{P}}[w_i|c]=\frac{\textrm{count}(w_i,c)}{\underbrace{\sum_{\textrm{All words }w}\textrm{count}(w,c)}_{\textrm{count}(c)}}" /></a>
  
  which is the fraction of word <a href="https://www.codecogs.com/eqnedit.php?latex=w_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_i" title="w_i" /></a> appearing among all words of all the documents of class `c`.
  
  * Smoothing:
  
  If a word `w` in a test document doesn't appear in any training document of class `c`, rather than setting <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{P}[w|c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{P}[w|c]" title="\mathbb{P}[w|c]" /></a> equal to `0`, do smoothing:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{\mathbb{P}}[w|c]=\frac{\textrm{count}(w,c)&plus;m}{\textrm{count}(c)&plus;mV}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{\mathbb{P}}[w|c]=\frac{\textrm{count}(w,c)&plus;m}{\textrm{count}(c)&plus;mV}" title="\widehat{\mathbb{P}}[w|c]=\frac{\textrm{count}(w,c)+m}{\textrm{count}(c)+mV}" /></a>
  
  where `V` is the vocabulary size, i.e., number of all words in the training documents.
  
  __Type2__ 
  
   In this case, the number of features of for any document is the number of words in the vocabulary. Each feature is binary, having value 1 when the corresponding word appears in the document and value 0 otherwise (there's an implicit word order determined by the way the vocabulary has been stored).

<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{\mathbb{P}}[w_i|c]=\frac{\textrm{\&hash;&space;documents&space;of&space;class&space;}&space;c&space;\textrm{&space;containing&space;}&space;w_i&plus;m}{\textrm{\&hash;&space;documents&space;of&space;class&space;}&space;c&space;&plus;mV}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{\mathbb{P}}[w_i|c]=\frac{\textrm{\&hash;&space;documents&space;of&space;class&space;}&space;c&space;\textrm{&space;containing&space;}&space;w_i&plus;m}{\textrm{\&hash;&space;documents&space;of&space;class&space;}&space;c&space;&plus;mV}" title="\widehat{\mathbb{P}}[w_i|c]=\frac{\textrm{\# documents of class } c \textrm{ containing } w_i+m}{\textrm{\# documents of class } c +mV}" /></a>

where `V=2` corresponding to binary features in this case. The formula above, accounts for smoothing.

## Procedure

### Learning

 1. From training data extract the vocabulary (Type I) and consequently the number of features (Type II).
 
 2. For every class `c`, compute <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{\mathbb{P}}[c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{\mathbb{P}}[c]" title="\widehat{\mathbb{P}}[c]" /></a>.
 
 3. Compute <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{\mathbb{P}}[w_i|c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{\mathbb{P}}[w_i|c]" title="\widehat{\mathbb{P}}[w_i|c]" /></a> according to either Type I or Type II.

## Remarks

 * Naive Bayes is fast ( just count words) and has low storage requirements. 
 * It's robust to irrelevant features since they cancel each other. If a feature `w` is irrelevant, <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{\mathbb{P}}[w|c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{\mathbb{P}}[w|c]" title="\widehat{\mathbb{P}}[w|c]" /></a> becomes almost unifrom.

---
---

We will be using a subset of the [20 newsgroups data set](http://qwone.com/~jason/20Newsgroups/) that has already been pre-processed (we convert to lowercase in addition and only consider word tokens that are alphbaet-only, i.e., `token.isalpha() = True`). The data files are in the `imbmac` directory. More specically, each dataset for this project includes text documents containing articles from two different categories. The task is to identify the category of each document. The dataset includes articles that were posted on newsgroups **comp.sys.mac.hardware** and **comp.sys.ibm.pc.hardware**. Each document is a message posted to one of the groups. Our class labels Yes and No capture membership in these groups (Yes corresponds to ibm).

Goal: Learn a classifier that, when given a new article, assigns it to the correct group.

---
[//]: # (The script `learning_curves.m` generates 4 learning curves for ibmmac dataset for each variant of Baive-Bayes, Type1 and Type2, without smoothing, m=0, and with Laplace Smoothing, m=1. This produces 4 plots of accuracy on the test set versus 10 different training set sizes 0.1N, 0.2N, ..., 0.9N, and N.)
