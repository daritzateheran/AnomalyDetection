from operator import concat
from flask import Flask, request, g
from keras.models import load_model
import ast
import numpy as np
import time
import pymysql, os


app = Flask(__name__)

# def StandarScaler_(file):
#   df = pd.read_csv(file)
#   #media xyz = 4.432655734498486  -1.2222901089734424 -0.9580775654058441
#   #varianza xyz = 7.606028685537751  8.55897071763424 6.497116567848398
#   df['x']=(df['x']-s.mean(df['x']))/s.stdev(df['x'])
#   df['y']=(df['y']-s.mean(df['y']))/s.stdev(df['y'])
#   df['z']=(df['z']-s.mean(df['z']))/s.stdev(df['z'])
#   return df


def get_conn():
    if "conn" not in g:
        g.conn = pymysql.connect(
            host=os.getenv('FLASK_DATABASE_HOST'),
            user=os.getenv('FLASK_DATABASE_USER'),
            password=os.getenv('FLASK_DATABASE_PASSWORD'),
            database=os.getenv('FLASK_DATABASE')
        )
        g.cur=g.conn.cursor()
    return g.conn, g.cur

labels = ["ANORMAL","NORMAL"]
model=load_model('../modelos/Model_swr_RawData.h5')
#model=load_model('/home/ubuntu/AnomalyDetection/modelos/Model_swr_RawData.h5')

@app.route("/sign", methods=["POST"])
def sign():
    key=request.form['Key']
    conn, cur = get_conn()
    cur=conn.cursor()
    cur.execute(f"SELECT users.ID FROM AnomalyData.users WHERE users.id_key ='"+key+"'")
    conn.commit() #si lo quito no sirve
    datos = cur.fetchall()
    cur.close()
    return (str(datos[0][0]))

@app.route("/add_caregiver", methods=["POST"])
def add_caregiver():
    key=request.form['Key']
    name_cg=request.form['contact']
    phone=request.form['numero']
    conn, cur = get_conn()
    cur=conn.cursor()
    cur.execute(f"SELECT caregivers.phone, caregivers.keyUser FROM AnomalyData.caregivers WHERE phone='"+phone+"' and keyUser='"+key+"'")
    conn.commit() #si lo quito no sirve
    datos=cur.fetchall()

    if(datos):
        cur.close()
        print("Caregiver already registered")
        return ("Caregiver already registered")
    else:
        cur.execute(f"INSERT INTO AnomalyData.caregivers (name_caregiver, phone, keyUser) VALUES (%s,%s,%s)",(name_cg, phone, key))
        conn.commit() #si lo quito no sirve
        cur.close()
        print("Contact added successfully")
        return ("Contact added successfully")

@app.route("/loadCaregivers", methods=["POST"])
def loadCaregivers():
    key=request.form['Key']
    conn, cur = get_conn()
    cur=conn.cursor()
    cur.execute(f"SELECT  caregivers.phone FROM AnomalyData.caregivers WHERE caregivers.keyUser='"+key+"'")
    conn.commit() #si lo quito no sirve
    datos = cur.fetchall()
    datos = list(datos)
    print(datos.split(','))
    cur.close()
    return (datos.split(','))

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

@app.route('/changes', methods=["POST","GET"]) #git hub
def pull():
    os.system('cd /home/ubuntu/AnomalyDetection/Server && git reset --hard && git pull') #esta linea de codigo hace que sea automatico el cambio del codigo si todas las instancias estan prendidas
    return 'hello'

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3000)