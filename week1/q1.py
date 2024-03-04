# Write a function  count_vowels(word) that takes a word as an argument and returns the number of vowels in the word

def count_vowels(word):
    vowels = ['a','e','i','o','u']
    count = 0
    for elem in word:
        if (elem in vowels):
            count+=1
    return count
