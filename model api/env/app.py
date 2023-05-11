from flask import Flask, jsonify, request, make_response
from mongoscripts import *
import numpy as np
import pandas as pd
from trainer import SGD

app = Flask(__name__)

@app.route("/recommend", methods=['POST','GET'])
def recommend():
    user_id = request.args.get('user_id')
    data = getRatings(user_id)
    dataRating = data["ratings"]
    dataCount = data["count"]
    if dataCount > 4:
        df = pd.DataFrame(dataRating)
        ratingData = np.zeros(45)
        for i in range(df.shape[0]):
            ind = int(df.iloc[i,0].split('-')[1])
            ratingData[ind] = df.iloc[i,1]
        prediction_matrix = model.train_new_user(ratingData)
        prediction_df = pd.DataFrame(prediction_matrix)
        prediction_df = prediction_df.rename(columns={0:"Predicted_ratings"})
        prediction_df = prediction_df.sort_values(by="Predicted_ratings",ascending=False).reset_index()
        top15 = prediction_df.head(15)
        fIds = np.random.choice(top15["index"].values, 9, replace=False)
        futsalNames = []
        for fid in fIds:
            futsalNames.append(trainData.index.values[fid])
    else:
        futsalNames = np.random.choice(top20futsals["Futsal_Name"].values, 9, replace=False)
    response = getFutsalInfos(list(futsalNames))
    return make_response(response, 201)

if __name__ == '__main__':
    try:
        model = pickle.load(open('recomm-model.pkl','rb'))
        trainData = pickle.load(open('trainData.pkl','rb'))
        top20futsals = pickle.load(open('top20futsals.pkl', 'rb'))
        print("API up and running...")
        app.run(debug=True)   
    except Exception as e:
        print("Error while loading model or running the application")
        print(e)