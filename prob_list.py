

def mixing_prob_controller_test(cfg):
    prob_list = []
    mixing_prob = 0
    min_prob = 0.1
    base_prob = 0.95
    for i in range(700):
        mixing_prob = base_prob * (0.98 ** i)
        if mixing_prob > 0.1:
            prob_list.append(mixing_prob)
        else:
            prob_list.append(min_prob)
    return prob_list

def mixing_prob_controller_test2(cfg):
    prob_list = []
    mixing_prob = 0
    min_prob = 0.1
    base_prob = 0.6

    for i in range(80):
        prob_list.append(0.6)
    for i in range(40):
        prob_list.append(0.4)  
    for i in range(40):
        prob_list.append(0.3)
    for i in range(40):
        prob_list.append(0.2)
    for i in range(400):
        prob_list.append(0.1)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test3(cfg):
    prob_list = []
    mixing_prob = 0
    min_prob = 0.1
    base_prob = 0.6

    for i in range(60):
        prob_list.append(0.6)
    for i in range(40):
        prob_list.append(0.4)  
    for i in range(40):
        prob_list.append(0.3)
    for i in range(40):
        prob_list.append(0.2)
    for i in range(400):
        prob_list.append(0.1)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test4(cfg):
    prob_list = []
    mixing_prob = 0
    min_prob = 0.1
    base_prob = 0.6

    for i in range(40):
        prob_list.append(1.0)
    for i in range(40):
        prob_list.append(0.8)  
    for i in range(40):
        prob_list.append(0.6)
    for i in range(40):
        prob_list.append(0.4)
    for i in range(40):
        prob_list.append(0.2)
    for i in range(200):
        prob_list.append(0.1)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test5(cfg):
    prob_list = []

    for i in range(400):
        tmp = 0.992 ** i
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test6(cfg):
    prob_list = []

    for i in range(400):
        tmp = 0.993 ** i
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test7(cfg):
    prob_list = []
    
    for i in range(50):
        tmp = 1.0
        prob_list.append(tmp)

    for i in range(400):
        tmp = 0.993 ** i
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test7(cfg):
    prob_list = []
    
    for i in range(50):
        tmp = 1.0
        prob_list.append(tmp)

    for i in range(400):
        tmp = 0.993 ** i
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test8(cfg):
    prob_list = []
    
    for i in range(40):
        tmp = 1.0
        prob_list.append(tmp)

    for i in range(400):
        tmp = 0.995 ** i
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test9(cfg):
    prob_list = []
    
    for i in range(40):
        tmp = 1.0
        prob_list.append(tmp)

    for i in range(400):
        tmp = 0.996 ** i
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list


def mixing_prob_controller_test10(cfg):
    prob_list = []
    
    for i in range(40):
        tmp = 1.0
        prob_list.append(tmp)

    for i in range(400):
        tmp = 0.996 ** i
        if tmp < 0.5:
            tmp = 0.5
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test11(cfg):
    prob_list = []
    
    for i in range(40):
        tmp = 1.0
        prob_list.append(tmp)

    for i in range(400):
        tmp = 0.997 ** i
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list

def mixing_prob_controller_test12(cfg):
    prob_list = []
    
    for i in range(40):
        tmp = 1.0
        prob_list.append(tmp)

    for i in range(400):
        tmp = 0.9985 ** i
        prob_list.append(tmp)
    
    print('prob')
    print(*prob_list)

    return prob_list