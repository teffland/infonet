from gensim.models import word2vec

print "Loading..."
model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print "Exporting to txt..."
model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)
