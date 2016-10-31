
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
