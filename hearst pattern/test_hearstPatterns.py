import unittest
from hearstPatterns import HearstPatterns
from collections import defaultdict
import re
from tqdm import tqdm 

class TestHearstPatterns(unittest.TestCase):

    def test_hyponym_finder(self):
        h = HearstPatterns(extended=True)

        file = open("a.txt",'r')
        line1 = file.readlines()
        file.close()

        line = ""
        for i in range(len(line1)):
            text = line1[i]
            text = text.replace("\n"," ")
            line += text

        #split sentences on .,?,!
        line2 = ""
        for i in range(len(line)):
            if line[i] in {'.','?','!'}:
                line2 += line[i]
                line2 += "\n"
            else:
                line2 += line[i]

        line = "".join(line2)
        line = line.split("\n")
        #print(type(line))
        dict = defaultdict(set)

        print(len(line))
        # del line[201622]
        jk = 0
        for i in tqdm(line):
            # jk += 1
            # if jk == 200000:
            #     break
            # i = i.lstrip(" ")
            # # print("=",i,"=")
            # i = re.sub(r'[^\w\s]', '', i)
            # i = re.sub(r'[0-9]', '', i)
            # # print(i)
            hyps1 = h.find_hyponyms(i)
            print(hyps1)
            for ii,jj in hyps1:
                dict[ii].add(jj)

        hypo = open("hypo.txt","a")
        hyper = open("hyper.txt","a")

        for i in dict:
            hypo.write(i+"\n")
            data = ""
            for j in dict[i]:
                data += j + "\t"
            hyper.write(data+"\n")
        hypo.close()
        hyper.close()


       


if __name__ == '__main__':
    unittest.main()