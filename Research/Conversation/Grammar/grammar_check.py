import nltk

def check(input):
    wrong_syntax=1
    sent_split = input.split()

    rd_parser = nltk.RecursiveDescentParser(nltk.data.load('Research/Language/Grammar/english_grammer.cfg'))
    for tree_struc in rd_parser.parse(sent_split):
        s = tree_struc
        wrong_syntax=0
        print("Correct Grammer !!!")
    if wrong_syntax==1:
        print("Wrong Grammer!!!!")

if __name__ == "__main__":
    text_input = "the good child walked happily "
    
    check(text_input)