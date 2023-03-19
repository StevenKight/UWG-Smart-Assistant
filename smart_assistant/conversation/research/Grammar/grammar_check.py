import nltk

def grammar_check(input):
    wrong_syntax=1
    # sent_split = input.split()

    text = nltk.word_tokenize(input)

    tagged = nltk.pos_tag(text)
    print(tagged)
    tags = [tag[1] for tag in tagged]

    rd_parser = nltk.RecursiveDescentParser(nltk.data.load('smart_assistant/conversation/research/Grammar/english_grammer.cfg'))

    for _ in rd_parser.parse(tags):
        wrong_syntax=0

    if wrong_syntax==1:
        return False
    
    return True

if __name__ == "__main__":
    text_input = "the good dog walked fast"

    if grammar_check(text_input):
        print("Correct Syntax")
    else:
        print("Wrong Syntax")