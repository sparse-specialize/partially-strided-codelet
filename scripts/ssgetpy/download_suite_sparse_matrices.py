from query import search

result = search(nzbounds=(100000,100000000), isspd=False, limit=1000000000)
result.download(extract=True,destpath="/Users/kazem/development/codelet_mining/scripts/ssgetpy/mm/")
