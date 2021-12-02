import pymongo
from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://Oecophylla:rangga2203@cluster0.yjcmj.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = cluster["mask_detection"]
collection = db["default"]

# post = {"_id": 0, "name": "Oecophylla", "score": 5}

# collection.insert_one(post)

post1 = {"_id":5, "name":"Naya"}
post2 = {"_id":6, "name":"Cece"}

res = collection.delete_many({})

# post_count = collection.count_documents({})
# print(post_count )
# print(res)

# for r in res:
#     print(r)
