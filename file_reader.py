from collections import Counter
import re
import codecs


class FileReader(object):
    def __init__(self, filename, sentiment=None):
        # set the initial filename
        self.filename = filename
        # Set the sentiment
        self.sentiment = sentiment

        # Initialize empty dicts
        self.reviews = {}

    def get_review_list(self):
        reviewList = []
        for id, review in self.reviews.iteritems():
            reviewList.append(review)
        return reviewList

    def remove_punctuation(self, sentence):
        # Remove commas, full stops, semicolons, /, hyphens
        sentence = sentence.replace(",", " ")
        sentence = sentence.replace(";", " ")
        sentence = sentence.replace(";", " ")
        sentence = sentence.replace("/", " ")
        sentence = sentence.replace("-", " ")
        sentence = sentence.replace("!", " ")
        return sentence

    def parse_file(self):
        self.reviews = {}
        self.word_count = 0
        # Parse the file and create the dict
        if self.filename != None:
            with codecs.open(self.filename, "rb", encoding='utf8') as training_file:
                str = ""
                for line in training_file:
                    # Get ID
                    split_array = line.split()
                    id = split_array[0]
                    review = " ".join(split_array[1:])
                    review = self.remove_punctuation(review)
                    self.reviews[id] = { "review": review, "id": id, "sentiment": self.sentiment}
                    str += review
                words = re.findall(r'\w+', str)
                self.word_count += len(words)
                cnt = Counter(words)
            return cnt
