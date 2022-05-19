from flask import Flask, request
from keras.models import load_model
import ast
import numpy as np
import time


app = Flask(__name__)

# def StandarScaler_(file):
#   df = pd.read_csv(file)
#   #media xyz = 4.432655734498486  -1.2222901089734424 -0.9580775654058441
#   #varianza xyz = 7.606028685537751  8.55897071763424 6.497116567848398
#   df['x']=(df['x']-s.mean(df['x']))/s.stdev(df['x'])
#   df['y']=(df['y']-s.mean(df['y']))/s.stdev(df['y'])
#   df['z']=(df['z']-s.mean(df['z']))/s.stdev(df['z'])
#   return df

labels = ["ANORMAL","NORMAL"]
model=load_model('modelos/Model_swr_RawData.h5')

@app.route("/post", methods=["POST"])
def model_():
    t=time.time() 
    value=request.form['value']
    timestamp=request.form['timestamp']
    vector_150=ast.literal_eval(value)
    segment=np.reshape(np.array(vector_150), (1,50,3))
     
    predict = model.predict(segment)
    predict = labels[np.argmax(predict[0])]

    print("Time = ", time.time() - t)
    print(predict + " TIMESTAMP = " + str(timestamp))
    return (predict + " TIMESTAMP = " + str(timestamp))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3000)