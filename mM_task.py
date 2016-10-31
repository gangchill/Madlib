import sys
with open('KLSadd.tex', "r") as old, open('KLSadd_New.tex', "w") as new:
    for line in old:
        words = []
        print 'no mM swap :', line
        for word in line:
            # print 'before mM swap :', word
            letters = []
            for letter in word:
                if letter == 'm' or letter == 'M':
                    letter = letter.swapcase()
                else:
                    pass
                letters.append(letter)
            # print 'after mM swap :', ''.join(letters)
            words.append(''.join(letters))
        print 'do mM swap :', ''.join(words)
        new.write(''.join(words))
#
# f = open('KLSadd.tex', "r")
# line = f.readline()
# while line:
#     print 'before mM swap :', line
#     list1 = []
#     for letter in line:
#         if letter == 'm' or letter == 'M':
#             letter = letter.swapcase()
#         else:
#             pass
#         list1.append(letter)
#     print 'after mM swap :', ''.join(list1)
#     line = f.readline()
# f.close()
# print "Finished!"
#
#
#
# def print_words(file_name):
#     input_file = open(file_name, 'r')
#     lines = input_file.readlines()
#     lines = ' '.join(x for x in lines)
#     print(lines)
#
# if __name__ == '__main__':
#     print_words(sys.argv[1])
