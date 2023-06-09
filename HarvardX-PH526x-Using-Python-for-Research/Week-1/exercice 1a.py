import string
alphabet = string.ascii_letters

"""

The lower and upper cases of the English alphabet is stored as alphabet.
Consider the sentence
'Jim quickly realized that the beautiful gowns are expensive'.
Create a dictionary count_letters with keys consisting of each unique
letter in the sentence and values consisting of the number of times each
letter is used in this sentence.
Count both upper case and lower case letters separately in the dictionary.

"""

sentence = 'Jim quickly realized that the beautiful gowns are expensive'
#Create an empty dictionnary
count_letters = {}
#Loop over the dictionnary
for letter in sentence: 
    #iterates through every letter in sentence
    
    if letter in alphabet: 
        #checks whether letter in in alphabet
        
        if letter in count_letters: 
            #if letter is in the alphabet, it further checks whether it is already in the dictionary
            
            count_letters[letter] += 1 
            #If so, it's count value is incremented by 1

        else:
            count_letters[letter] = 1 
            #if the letter is not in the dictionary, its value is set to 1
