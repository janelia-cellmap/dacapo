# Released into the Public Domain by fpgaminer@bitcoin-mining.com


import hashlib
import math


def load_dictionary (dictionary_file=None):
	if dictionary_file is None:
		dictionary_file = "words.txt"

	with open (dictionary_file, 'rb') as f:
		dictionary = f.read ().splitlines ()
	
	return dictionary


def default_hasher (data):
    hash = hashlib.sha256()
    hash.update(data)
    return hash.hexdigest()


def hash_phrase (data, num_words=1, dictionary=None, hashfunc=default_hasher):
	# Dictionary
	if dictionary is None:
		dictionary = load_dictionary ()
	
	dict_len = len (dictionary)
	entropy_per_word = math.log (dict_len, 2)

	# Hash the data and convert to a big integer (converts as Big Endian)
	hash = hashfunc (data)
	available_entropy = len (hash) * 4
	hash = int (hash, 16)

	# Check entropy
	if num_words * entropy_per_word > available_entropy:
		raise Exception ("The output entropy of the specified hashfunc (%d) is too small." % available_entropy)

	# Generate phrase
	phrase = []

	for i in range (num_words):
		remainder = hash % dict_len
		hash = hash // dict_len

		phrase.append (dictionary[remainder])
	
	return " ".join (phrase)
