import string
from collections import Counter
alphabet = string.ascii_letters.lower()
sentence = 'Jim quickly realized that the beautiful gowns are expensive'
counter_letters = {}
for letter in sentence:
    if letter in alphabet:
        if letter in counter_letters:
            counter_letters[letter] += 1
        else:
            counter_letters[letter] = 1

def counter(sentence):
    alphabet = ascii_letters
    counter_letters = {}
    for letter in sentence:
        if letter in alphabet:
            if letter in counter_letters:
                counter_letters[letter] += 1
            else:
                counter_letters[letter] = 1
    return counter_letters
print(counter_letters)


address = """Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth."""

def counter(address):
    address_count = {}
    for letter in address:
        if letter in address_count:
            address_count[letter] += 1
        else:
            address_count[letter] = 1
    return address_count
print(address_count)




from func import counter #imports counter function, previously made and saved as func.py
from string import ascii_letters

address_count = {}

with open("address.txt") as f:

    address = [line.rstrip('\n') for line in f]

count = 0

while count < len(address):

    taddress_count = counter(address[count])

    for key in taddress_count.keys():

        if key in address_count.keys():

            taddress_count[key] += address_count[key]
        
    count += 1
    address_count.update(taddress_count)

print(address_count)




