from scipy.sparse import diags

print(diags([1,1,0,1], shape=(4, 4)).toarray())
