import sys
import math
import numpy as np
import csv

class Tree:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key
        self.max = None
    
args = sys.argv
#input = open(args[1], 'r')
train = open(args[1], 'r')

test = open(args[2], 'r')

max_depth = args[3]
max_depth = int(max_depth)

train_out = args[4]
test_out = args[5]
metrics_out = args[6]

train_table = list(csv.reader(train, delimiter='\t'))
train_table = np.array(train_table)

test_table = list(csv.reader(test, delimiter='\t'))
test_table = np.array(test_table)
#print(table)

attributes = np.unique(train_table[1:len(train_table),len(train_table[0])-1])
#print(attributes)
attribute1 = attributes[0]
attribute2 = attributes[1]
train_l = []
test_l = []


def entropy(table, col_num):
    total = len(table)-1
    if table.size == 0:
        return 0
    t = table[1:len(table),col_num]
    if t.size == 0 or t.size == 1:
        return 0
    #print(t)
    val1 = t[0]
    val1_count = 0
    for e in t:
        if e == val1:
            val1_count += 1
    val2_count = total - val1_count
    p_val1 = val1_count/total
    p_val2 = val2_count/total
    if p_val1 == 0 and p_val2 == 0:
        entropy = 0
    elif p_val1 == 0:
        entropy = -(p_val2)*np.log2(p_val2)
    elif p_val2 == 0:
        entropy = -(p_val1)*np.log2(p_val1)
    else: 
        entropy = -((p_val1)*np.log2(p_val1) + (p_val2)*np.log2(p_val2))

    return entropy

def mutual_information(table):
    total = len(table)-1
    cols = len(table[0])-1
    Hy = entropy(table, cols)
    mi_list = []
    for i in range(cols):
        #print(i)
        attribute = table[0][i]
       # print(table.size)
        val1 = table[1][i]
        split = table[:,i] == val1
        split1 = np.vstack([table[0], table[split]])
        split2 = table[~split]
        p_split1 = (len(split1)-1)/total
        p_split2 = (len(split2)-1)/total
        MI = Hy - p_split1*entropy(split1,cols) - p_split2*entropy(split2,cols)
        mi_list.append(MI)
    print(mi_list)
    return mi_list

def construct_tree(depth, table):
    #if depth < 2:
    if depth < max_depth and depth < len(table[0])-1 and np.shape(table)[0] != 1:
        mi_list = mutual_information(table)
        
        if all([ v == 0 for v in mi_list ]):
            return
        #print(mi_list)
        #print(mi_list)
        split_index = mi_list.index(max(mi_list))
        #print(split_index)
        #val1 = table[1][split_index]
        x1_a = np.sort(np.unique(table[1:len(table),split_index]))
        #print(x1_a)
        val1 = x1_a[0]
        split = table[:, split_index] == val1
        
        split1 = np.vstack([table[0], table[split]])
        split2 = table[~split]

        breakdown1 = stump(split1)[0]
        max_val_l = stump(split1)[1]
        breakdown2 = stump(split2)[0]
        max_val_r = stump(split2)[1]

        node = Tree(table[0][split_index])
        
        d = dict()
        
        d[x1_a[0]] = max_val_l
        d[x1_a[1]] = max_val_r

        node.max = d
        #print(node.max)
        depth_str = (depth+1)*'|'
        
        print(f'{depth_str} {node.val} = {x1_a[0]} : {breakdown1}')
        
        node.left = construct_tree(depth+1,split1)
        #print(node.max)

        print(f'{depth_str} {node.val} = {x1_a[1]} : {breakdown2}')
        
        node.right = construct_tree(depth+1,split2)
        
        #print(node.max)

        return node
    else:

        return None

def stump(table):
    cols = len(table[0])-1
    last_column =  table[1:len(table),cols]
    val1_count = 0
    val2_count = 0
    for e in last_column:
        if e == attribute1:
            val1_count += 1
        elif e == attribute2:
            val2_count += 1
    result_str = f'[{val1_count} {attribute1}/{val2_count} {attribute2}]'
    if val1_count > val2_count:
        max_attribute = attribute1
    else:
        max_attribute = attribute2

    return (result_str, max_attribute)


def leaf_values(root):
    if root.left==None and root.right==None:
            print(root.max)
        #First recur the left child
    if root.left:
        leaf_values(root.left)
        #Recur the right child at last
    if root.right:
        leaf_values(root.right)

def classify_row(L, header, tree, row):
    head = header.tolist()
    index = head.index(tree.val)
    #print(tree.val)
    key_list = []
    for key in tree.max:
        key_list.append(key)
    key_list.sort()
    #print(row)
    #print(key_list)
    if row[index] == key_list[0]:
        if tree.left:
            #print('entered here')
            classify_row(L, header, tree.left, row)
        

        else:
           
            L.append(tree.max[key_list[0]])
            return tree.max[key_list[0]]
        
    else:
        if tree.right:
            
            classify_row(L, header, tree.right, row)
      
        
        else:
            #print(tree.max[key_list[1]])
            L.append(tree.max[key_list[1]])
            return tree.max[key_list[1]]
    
def classify_table(L, tree, table):
    header = table[0]
    l = []
    i = 0
    for row in table[1:len(table)]:
        i += 1
        target = classify_row(L, header, tree, row)
        
        l.append(target)
        
    return l

def calc_metrics(table, l):
    wrong_answers = 0
    i = 0
    for row in range(1,len(table)):
        if table[row][len(table[row])-1] != l[i]:
            wrong_answers += 1
        i += 1
    return wrong_answers/(len(table)-1)

print(stump(train_table)[0])
train_tree = construct_tree(0, train_table)
#test_tree = construct_tree(0, test_table)

classify_table(train_l, train_tree, train_table)
classify_table(test_l, train_tree, test_table)


train_error = calc_metrics(train_table, train_l)
test_error = calc_metrics(test_table, test_l)

def write_labels(s, l):
    with open(s, 'w') as f_out:
        for line in l:
            f_out.write(f'{line}\n')

write_labels(train_out, train_l)
write_labels(test_out, test_l)


print(train_error)
print(test_error)

with open(metrics_out, 'w') as f_out:
    f_out.write(f'error(train): {train_error}\nerror(test): {test_error}')