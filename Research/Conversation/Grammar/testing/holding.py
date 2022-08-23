import nltk

load_grammar = nltk.data.load('file:Englishgrammer.cfg')
file_input = ["the child walked happily"]
for sent in file_input:
    wrong_syntax=1
    sent_split = sent.split()
    print("\n\n"+ sent)
    rd_parser = nltk.RecursiveDescentParser(load_grammar)
    for tree_struc in rd_parser.parse(sent_split):
        s = tree_struc
        wrong_syntax=0
        print("Correct Grammer !!!")
        print(str(s))
        f = open("demoEnglish.txt", "a")
        f.write("Correct Grammer!!!!!")
        f.write(str(s))
        f.close()
    if wrong_syntax==1:
        print("Wrong Grammer!!!!!!")
        f = open("demoEnglish.txt", "a")
        f.write("Wrong Grammer!!!!!")
        f.close()