import sys
test = ''
sys.stdout = open('TMP.txt', 'w')
for i in range(10):
    test = str(i)
    print(test)