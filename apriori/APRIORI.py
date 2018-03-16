import numpy as np


# Read all data as a binary matrix where each row is a basket and
#    each column are a specific item.
# Returns a binary matrix and a list of the items in the same order as the columns.
def read_basket_data(file_name = "data.txt"):

    with open(file_name) as f:
        lines = [x.lower().strip().split(",") for x in f.readlines()]
        
    items = list(set([item for basket in lines for item in basket]))
    items.sort()

    return np.asarray([[1 if i in l else 0 for i in items] for l in lines]) , np.asanyarray(items)
    

# HELPER FUNCTIONS
# Print out a given rule
def print_rule(names, left, right, confidence):
    
    print("{0} -> {1} with confidence: {2}".format(
            ", ".join(names[left == 1]),
            ", ".join(names[right == 1]),
            confidence))

def check_item(item, column):
    for i in range(0, len(column)):
        if (item == column[i]).all():
            return True
    return False

def check_rule(l, r, lhs, rhs):
    for i in range(0, lhs.shape[0]):
        if (np.logical_or(l, r) == np.logical_or(lhs[i], rhs[i])).all():
            return True
    return False


# This function should implement the calculation of the support.
# That is how many times the itemset x occures in X.
def calculate_support(x, X):
    support = 0.0
    for i in range(0, X.shape[0]):
        if (np.logical_and(x, X[i]) == x).all():
            support = support + 1
    return support / X.shape[0]


# This function should implement the calculation of confidence.
def calculate_confidence(x, y, X):
    transactions = 0.0
    association = 0.0
    for i in range(0, X.shape[0]):
        if (np.logical_and(x, X[i]) == x).all():
            transactions = transactions + 1
            if (np.logical_and(y, X[i]) == y).all():
                association = association + 1
    return association / transactions


# This function should implement apriori frequent itemset generation
# You should return binary representation of all frequent itemsets in the dataset. 
# That is, all itemsets that has a support greater than min_support
# You should also return the support for each returned itemset
def generate_frequent(X, min_support):

    def generate_k1(X, min_support):
            zeros = np.zeros(X.shape[1], dtype=np.int)
            itemset = np.empty([0, X.shape[1]], dtype=np.int)
            support = np.empty([0, 1], dtype=np.float)

            for i in range(0, X.shape[1]):
                k1_itemset = zeros.copy()
                k1_itemset[i] = 1
                k1_support = calculate_support(k1_itemset, X)
                if k1_support > min_support:
                    itemset = np.vstack((itemset, k1_itemset))
                    support = np.vstack((support, k1_support))
            return itemset, support

    def generate_kx(X, k, min_support):
        support = np.empty([0, 1], dtype=np.float)
        kx_itemset = np.empty([0, k.shape[1]], dtype=np.int)
        for i in range(0, k.shape[0]):
            some_item = k[i]
            for j in range(0, k.shape[0]):
                if (i == j):
                    continue
                some_other_item = k[j]
                new_item = np.logical_or(some_item, some_other_item).astype(np.int)
                new_support = calculate_support(new_item, X)
                if new_support > min_support and check_item(new_item, kx_itemset) != True:
                    kx_itemset = np.vstack((kx_itemset, new_item))
                    support = np.vstack((support, new_support))
        return kx_itemset, support

    k1itemset, k1support = generate_k1(X, min_support)
    itemset = k1itemset
    support = k1support
    while True:
        kxItemset, kxSupport = generate_kx(X, k1itemset, min_support)
        if kxItemset.size == 0 or kxSupport.size == 0:
            return itemset, support
        k1itemset = kxItemset
        k1support = kxSupport
        itemset = np.vstack((itemset, k1itemset))
        support = np.vstack((support, k1support))


# This function should implement apriori rule generation based on the frequent itemsets
# The result should be a number of association rules, where each row in left and right represent 
# the left and right hand sides of an association rule (1 where an item is part of the left or right and
# 0 where the item is not part). 
# You should also return the confidence for each rule in c
def generate_rules(X, itemset, min_confidence):
    lhs = np.empty([0, X.shape[1]], dtype=np.int)
    rhs = np.empty([0, X.shape[1]], dtype=np.int)
    confidence = np.empty([0, 1], dtype=np.float)
    for i in range(0, itemset.shape[0]):
        antecedent = itemset[i]
        for j in range(0, itemset.shape[0]):
            consequent = itemset[j]
            if not ( (antecedent == consequent).all() or (np.logical_and(antecedent, consequent) == consequent).all() or check_rule(antecedent, consequent, lhs, rhs)):
                conf = calculate_confidence(antecedent, consequent, X)
                if conf > min_confidence:
                    lhs = np.vstack((lhs, antecedent))
                    rhs = np.vstack((rhs, consequent))
                    confidence = np.vstack((confidence, conf))
    return lhs, rhs, confidence

# The main program
# Run this when you have implemented all other parts.
def run_apropri():
    
    #Load data (binary representation for each transaction)
    print("Loading data...")
    X, names = read_basket_data()

    # Generate frequent itemsets
    print("Now generating itemsets...")
    min_support = 0.015
    itemsets, support = generate_frequent(X, min_support)
    print(str(len(itemsets)) + " itemsets were generated.")

    
    # Generate association rules
    print("Now generating association rules...")
    minConf = 0.6
    left, right, confidence = generate_rules(X, itemsets, minConf)
    
    
    # Print all rules
    print(str(len(left)) + " rules in total were found.")
    print("The following association rules were found:")
    for l,r,c in zip(left, right, confidence):
        print_rule(names, l, r, c)


run_apropri()