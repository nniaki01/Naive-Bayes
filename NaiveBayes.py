""" _________________________ Importing Module(s) __________________________"""
import math
from collections import defaultdict
from random import randint
import matplotlib.pyplot as plt
import numpy as np
""" ________________________________________________________________________"""

class NaiveBayes:
    def __init__(self, ds_name, portion, m):
        '''
        Args:
            ds_name(string): Name of the dataset and the corresponding folder;
            portion(float (0,1]): Portion of the training data used; to
            generate learning curves;
            m(float) : Smoothing parameter.
        '''    
        
        self.ds_name = ds_name  # e.g. "ibmmac"
        self.index_train = "index_train.txt"
        self.index_test = "index_test.txt"
        self.portion = portion
        self.m = m
        
        '''_______________________________ Type I _____________________________
        In this case, a given document has as many features as the number of
        words in it and a "token feature" has as many possible values as the
        words in the vocabulary (all words in training files).'''
        self.word_ct_dict_y = defaultdict(int) # {word: count given class yes}
        self.word_ct_dict_n = defaultdict(int) # {word: count given class no}
        self.total_word_ct_y = 0 # total count of word tokens in class yes
        self.total_word_ct_n = 0 # total count of word tokens in class no
        '''______________________________ Type II _____________________________
        In this case, the number of features of for any document is the number
        of words in the vocabulary. Each feature is binary, having value 1 when
        the corresponding word appears in the document and value 0 otherwise
        (there's an implicit word order determined by the way the vocabulary
        has been stored).'''
        self.word_docCt_dict_y = defaultdict(int) # {word: doc count given yes}
        self.word_docCt_dict_n = defaultdict(int) # {word: doc count given no}
        self.total_doc_ct_y = 0 # total number of documents in class yes
        self.total_doc_ct_n = 0 # total number of documents in class no
        ''' {feature: doc count based probability given yes}: '''
        self.feature_prob_y = defaultdict(float)
        ''' {feature: doc count based probability given no}: '''
        self.feature_prob_n = defaultdict(float) 
        
        self.vocab = set()
        self.vocab_y = set()
        self.vocab_n = set()
        self.test_docs = {} # {(filename, label): doc word list}
        self.total = 0      # total number of training docs

    def get_counts_given_fileLabel_train(self, filename, label):
        '''
        Args:
            filename(string): Filename of the training example document;
            label(yes/no)   : Label of the example document.
        '''    
        count = 0 # initialize word cnt of the given doc
        # have we counted cur doc for cur word:
        current_doc_flag_dict_y = defaultdict(int)
        current_doc_flag_dict_n = defaultdict(int)
        with open(filename, encoding = "ISO-8859-1") as f:
            for line in f: # read the file line by line
                words = line.split(" ")
                for word in words: # examine each word in each line
                    word = word.replace('\n', '')  # remove new line characters
                    if len(word) > 0:
                        count += 1
                        if label == "yes":
                            self.word_ct_dict_y[word] += 1
                            if not current_doc_flag_dict_y[word]:
                                self.word_docCt_dict_y[word] += 1
                                current_doc_flag_dict_y[word] = 1
                        else:
                            self.word_ct_dict_n[word] += 1
                            if not current_doc_flag_dict_n[word]:
                                self.word_docCt_dict_n[word] += 1
                                current_doc_flag_dict_n[word] = 1

        if label == "yes":
            self.total_doc_ct_y += 1
            self.total_word_ct_y += count
        else:
            self.total_doc_ct_n += 1
            self.total_word_ct_n += count
        # DEBUG: print("Number of |yes| documnets: ",self.total_doc_ct_y)
        # DEBUG: print("Number of |no| documnets: ",self.total_doc_ct_n)
        

    def derive_vocab(self):
        for word, cnt_ in self.word_ct_dict_y.items():
            self.vocab_y.add(word)
        for word, cnt_ in self.word_ct_dict_n.items():
            self.vocab_n.add(word)
        self.vocab = self.vocab_n.union(self.vocab_y)

    def get_type_II_probs(self):
        V = 2
        for feature in self.vocab:
                docCt_w_y = self.word_docCt_dict_y[feature]
                self.feature_prob_y[feature] = (1.0 * docCt_w_y + self.m)\
                    / (self.total_doc_ct_y + self.m * V)
                docCt_w_n = self.word_docCt_dict_n[feature]
                self.feature_prob_n[feature] = (1.0 * docCt_w_n + self.m)\
                    / (self.total_doc_ct_n + self.m * V)

    def get_text_given_fileLabel_test(self, filename, label):
        '''
        Args:
            filename(string): Filename of the test example document;
            label(yes/no)   : Label of the example document.
        '''
        doc_word_list = []
        with open(filename,  encoding = "ISO-8859-1") as f:
            for line in f:
                words = line.split(" ")
                for word in words:
                    word = word.replace('\n', '')  # remove new line characters
                    if len(word) > 0:
                        doc_word_list.append(word)

        self.test_docs[(filename, label)] = doc_word_list

    def process_index_file(self, index_file):
        '''
        Args:
            index_file: filename of the txt file containing example filenames
            and labels; ends in "_train" or "_test"
            e.g. 1|yes| means the example file 1.clean has label yes i.e., ibm
        '''

        ''' NOTE: If index_file ends in "_train" call
              self.get_counts_given_fileLabel_train(filename, label)
              else call 
              self.get_text_given_fileLabel_test (filename, label)
        '''
        if "train" in index_file:
            # Get the total number of training documents
            with open(index_file) as f:
                for N, l in enumerate(f):
                    pass
            N = N + 1
            N = int(N*self.portion)
            self.total = N
        count = 0 # initialize the processed doc count
        with open(index_file) as f:
            for line in f: # each line of format filename|label|
                filename_label = line.split("|")
                filename = filename_label[0] + ".clean"
                label = filename_label[1]
                if "train" in index_file:
                    # OPT: print("Processing", filename, "|", str(label))
                    self.get_counts_given_fileLabel_train(filename, label)
                else:
                    self.get_text_given_fileLabel_test(filename, label)

                # If doing learning curves stop once having processed the given
                # portion.
                count += 1
                if "train" in index_file and count == N:
                    break
                

    def train(self):
        self.process_index_file(self.index_train)
 
        self.derive_vocab()
        
        self.get_type_II_probs()
 
    def compute_class_scores(self, type_, doc_words):
        '''
        Args:
            type_(int): Naive Bayes variant
            doc_words(list): list of the words in the test documents
        '''
        m = self.m
        minus_inf = float("-inf")
        
        score_y = math.log((1.0 * self.total_doc_ct_y /self.total) + 10**(-10))
        score_n = math.log((1.0 * self.total_doc_ct_n /self.total) + 10**(-10))
        
        flag_y = 0 # have we hit -inf for |yes| score
        flag_n = 0 # have we hit -inf for |no| score

        if type_ ==1:
            '''____________________________ Type I ____________________________
            In this case, a given document has as many features as the number
            of words in it and a "token feature" has as many possible values as
            the words in the vocabulary (all words in training files).'''
            V = len(self.vocab)
            for word in doc_words:
                if word not in self.vocab:
                    ''' If a word in a test document does not appear in the
                    training set at all, i.e., none of the classes, skip it!'''
                    continue
                if not flag_y:
                    count_w_y = self.word_ct_dict_y[word]
                    word_prob_y = (1.0 * count_w_y + m)\
                    / (self.total_word_ct_y + m * V)
                    if word_prob_y != 0 and score_y != minus_inf:
                        score_y += math.log(word_prob_y)
                    else:
                        score_y= minus_inf
                        flag_y = 1
                if not flag_n:
                    count_w_n = self.word_ct_dict_n[word]
                    word_prob_n = (1.0 * count_w_n + m)\
                    / (self.total_word_ct_n + m * V)
                    if word_prob_n != 0 and score_n != minus_inf:
                        score_n += math.log(word_prob_n)
                    else:
                        score_n = minus_inf
                        flag_n = 1
                # Exit early if we've hit -inf score for both classes!
                if flag_y == 1 and flag_n == 1:
                    return score_y,score_n

            '''___________________________ Type II ____________________________
            In this case, the number of features of for any document is the
            number of words in the vocabulary. Each feature is binary, having
            value 1 when the corresponding word appears in the document and
            value 0 otherwise (there's an implicit word order determined by the
            way the vocabulary has been stored).'''
        else:
            V = 2     
            for feature in self.vocab:
                if feature in doc_words:
                    if not flag_y:
                        if (self.feature_prob_y[feature] != 0 
                        and score_y != minus_inf):
                            score_y += math.log(self.feature_prob_y[feature])
                        else:
                            score_y = minus_inf
                            flag_y = 1
                    if not flag_n:
                        if (self.feature_prob_n[feature] != 0 
                            and score_n != minus_inf):
                            score_n += math.log(self.feature_prob_n[feature])
                        else:
                            score_n = minus_inf
                            flag_n = 1
                else:
                    if not flag_y:
                        if ((1 - self.feature_prob_y[feature]) != 0 
                        and score_y != minus_inf):
                            score_y += math.log(1 - 
                                                self.feature_prob_y[feature])
                        else:
                            score_y = minus_inf
                            flag_y = 1
                    if not flag_n:
                        if ((1 - self.feature_prob_n[feature]) != 0 
                            and score_n != minus_inf):
                            # print(feature, "Debug: ", 1 - self.feature_prob_n[feature])
                            score_n += math.log(
                                    1 - self.feature_prob_n[feature])
                        else:
                            score_n= minus_inf
                            flag_n = 1
                # Exit early if we've hit -inf score for both classes!
                if flag_y == 1 and flag_n == 1:
                    return score_y,score_n

        return score_y,score_n

    def get_accuracy(self, type_):
        
        self.process_index_file(self.index_test)
        
        minus_inf = float("-inf")

        num_correct = 0.0
        test_count = 0
        for filenameLabel, doc_words in self.test_docs.items():
            
            label = filenameLabel[1]
            score_y, score_n = self.compute_class_scores(type_, doc_words)
            
            if score_y == minus_inf and score_n == minus_inf:
                num_correct += randint(0, 1)
                # OPT: print(filenameLabel[0], label, score_y, score_n, "Random")
            elif score_y == minus_inf and label == "no":
                num_correct += 1
                # OPT: print(filenameLabel[0], label, score_y, score_n, "True Negative")
            elif score_n == minus_inf and label == "yes":
                num_correct += 1
                # OPT: print(filenameLabel[0], label, score_y, score_n, "Ture Positive")
            elif score_y > score_n and label == "yes":
                num_correct += 1
                # OPT: print(filenameLabel[0], label, score_y, score_n, "Ture Positive")
            elif score_y < score_n and label == "no":
                num_correct += 1
                # OPT: print(filenameLabel[0], label, score_y, score_n, "True Negative")
            elif score_y == score_n:
                num_correct += randint(0, 1)
                # OPT: print(filenameLabel[0], label, score_y, score_n, "Random")
            # OPT: 
            """
            else:
                print(label, score_y, score_n, "False")
            """
            test_count += 1

        accuracy = 100 * num_correct / test_count

        # OPT: print('Accuracy = ',accuracy)
        return accuracy

# ___________________________________________________________________________ #
def main():
    '''
    Generating learning curves: Varying the number of training examples used,
    0.1*N, 0.2*N, ..., N, where N is the sizeof the available dataset
    '''
    learning_curves_values = defaultdict(list)
    ds_name = "ibmmac"
    print('Generating learning curves for smoothing parameters 0 and 1:')
    for m in range(2):
        for type_ in [1, 2]:
            for p in range(1,11):
                portion_ = p/10.0
                nb = NaiveBayes(ds_name, portion_, m)
                nb.train()
                stored_value = nb.get_accuracy(type_)
                print("m = ", str(m), " | Portion: ", portion_, " | Variant ",
                      type_, " | Accuracy: ", str(stored_value))
                learning_curves_values[(m,type_)].append(stored_value)

    '''
    Varying m
    '''
    varying_m = defaultdict(list)
    
    portion_ = 1
    m_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for type_ in [1, 2]:
        for m in m_list:
            nb = NaiveBayes(ds_name, portion_, m)
            nb.train()
            stored_value = nb.get_accuracy(type_)
            print("m = ", str(m), " | ", "Variant ", type_, "Accuracy: ",
                          str(stored_value))
            varying_m[type_].append(stored_value)
    
    return nb, learning_curves_values, varying_m, m_list

if __name__ == "__main__":
    nb_obj, learning_curves_values, varying_m, m_list = main()
    
    fig_1, ax = plt.subplots()
    
    line1, = ax.plot(np.linspace(0.1, 1, num=10, endpoint=True),
                     learning_curves_values[(0,1)], 'ro--', linewidth=2,
                      label = " m = 0 | Variant 1")
    
    line2, = ax.plot(np.linspace(0.1, 1, num=10, endpoint=True),
                      learning_curves_values[(0,2)], 'bo--', linewidth=2,
                      label = " m = 0 | Variant 2")
    
    line3, = ax.plot(np.linspace(0.1, 1, num=10, endpoint=True),
                     learning_curves_values[(1,1)], 'mo-', linewidth=2,
                      label = " m = 1 | Variant 1")
    
    line4, = ax.plot(np.linspace(0.1, 1, num=10, endpoint=True),
                      learning_curves_values[(1,2)], 'ko-', linewidth=2,
                      label = " m = 1 | Variant 2")

    legend = ax.legend(loc='upper left', shadow = False)
    plt.xlabel('Portion of the Available Data Used')
    plt.ylabel('Accuracy [%]')
    plt.title('Learning Curves')
    plt.show()    
    
    fig_2, ax_2 = plt.subplots()
    
    line5, = ax_2.plot(m_list, varying_m[1], 'bo-', linewidth=2,
                      label = "Variant 1")
    
    line6, = ax_2.plot(m_list, varying_m[2], 'ro-', linewidth=2,
                      label = "Variant 2")
    
    legend_2 = ax_2.legend(loc='upper left', shadow = False)
    plt.xlabel('Smoothing Paramter m')
    plt.ylabel('Accuracy [%]')
    plt.title('Varying m')
    plt.show()    