from flask import Flask, request, render_template
import sqlite3
from flask_socketio import SocketIO
import time
from sklearn.metrics import confusion_matrix
import datetime

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/transactions')
def transactions():
    return render_template('transactions.html', updateInterval=10000)


@app.route('/test')
def test():
    return render_template('test.html', updateInterval=10000)

@app.route('/transaction/<id>')
def transaction(id):
    data, description = get_person_data(id)
    print(str(datetime.datetime.fromtimestamp(data[1])))
    data[1] = str(datetime.datetime.fromtimestamp(data[1]))
    return render_template('transaction.html', data=data, description=description,
                           length=len(description))


def get_person_data(id):
    print(id)
    connection = sqlite3.connect('transactions.db')
    cursor = connection.cursor()
    cursor.execute(f'''SELECT *
                   FROM trs
                   WHERE ID={id}''')
    data = cursor.fetchall()
    description = [i[0] for i in cursor.description]
    connection.close()
    if data:
        return (list(data[0]), description)
    else:
        return None

@socketio.on('lazy_load_request')
def handle_lazy_load_request(args):
    data = get_data(args['last_id'])
    if data:
        socketio.emit('update_table', {'bottom_data': data, 'last_id':data[-1][0]})

@socketio.on('initialize_data')
def handle_lazy_load_request(args):
    data = get_data(limit=50)
    socketio.emit('update_table', {'bottom_data': data, 'last_id':data[-1][0], 'first_id':data[0][0]})
    if args.get('get_conf_m'):
        conf_m = get_conf_m()
        socketio.emit('update_conf_m', {'conf_m': conf_m})

@socketio.on('check_update')
def handle_update_load_request(args):
    data = get_data(args['first_id'], type='update')
    if data:
        socketio.emit('update_table', {'top_data': data, 'first_id':data[-1][0]})


# def add_conf_m(conf_m, data):


def get_conf_m():
    connection = sqlite3.connect('transactions.db')
    cursor = connection.cursor()
    cursor.execute(f'''SELECT target, predict
                   FROM trs''')
    data = cursor.fetchall()
    connection.close()
    y_true, y_pred = [], []
    for i in data:
        y_true.append(i[0])
        y_pred.append(i[1])
    conf_m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return conf_m.tolist()


def get_data(id_start=-1, limit=10, type='lazy'):
    connection = sqlite3.connect('transactions.db')
    cursor = connection.cursor()
    if type=='lazy':
        add_sql = f"WHERE id < {id_start}"
        order = "DESC"
    else:
        add_sql = f"WHERE id > {id_start}"
        order = "ASC"

    if id_start==-1:
        add_sql = ""
    # print(add_sql)
    cursor.execute(f'''SELECT id, Time, "from", "to", amount, target, predict
                   FROM trs
                   {add_sql}
                   ORDER BY time {order}
                   LIMIT {limit}''')
    all_data = cursor.fetchall()
    connection.close()
    return all_data

if __name__ == '__main__':
    # app.run()
    socketio.run(app, debug=True)


