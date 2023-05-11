from pymongo import MongoClient
import bson
import pickle

client = MongoClient('localhost', 27017)

db = client.get_database('projectTests')
userCollection = db.get_collection('user')
futsalCollection = db.get_collection('futsal')

futsals = pickle.load(open('futsals.pkl', 'rb'))

def getRatings(userId):
    ratings = userCollection.find_one({"_id": bson.ObjectId(userId)},{'ratings':1,'_id':0})
    count = len(ratings["ratings"])
    return {"count": count, "ratings": ratings["ratings"]}

def getFutsalInfos(futsalNames):
    futsalInfos = list(futsalCollection.find({"Name": {'$in': futsalNames}}))
    fInfos = {"futsalInfo": []}
    for info in futsalInfos:
        futsal = futsals[futsals["Futsal_Name"] == info["Name"]]
        info["_id"] = str(info["_id"])
        info["count"] = str(futsal["Count"].values)
        info["meanRating"] = str(futsal["Mean"].values)
        fInfos["futsalInfo"].append(info)
    return fInfos