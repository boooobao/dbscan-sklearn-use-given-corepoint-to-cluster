
def deal(filename):
    data = read(filename)
    data1 = []
    label = []
    for i in data:
        tmp = i.split(' ')
        if tmp[len(tmp)-1][-1]=="\n":
            tmp[len(tmp)-1] = tmp[len(tmp)-1][:-1]
        label.append(tmp[0])
        data1.append(tmp)
    return data1,label

def read_core_point(filename):
    f = open(filename,'r')
    tmp = f.readline()
    tmp = tmp.split(' ')
    tmp[1]=tmp[1][:-1]
    data = []
    print(tmp)
    for i in range(int(tmp[0]), int(tmp[1])):
        data.append(i)
    return data

def read(filename):
    f = open(filename, 'r')
    data = []
    while 1:
        line = f.readline()
        if not line:
            break
        data.append(line)

    f.close()
    return data

if __name__ == '__main__':
    deal('dataset/spiral.txt')