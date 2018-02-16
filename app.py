import os
from bottle import request, response
import bottle
from sklearn.externals import joblib

loaded_model = joblib.load('model/finalized_model.sav')
vectorizer = joblib.load('model/vectorizer.pk')

class EnableCors(object):

    name = 'enable_cors'
    api = 2

    def apply(self, fn, context):
        def _enable_cors(*args, **kwargs):
            # set CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

            if request.method != 'OPTIONS':
                # actual request; reply with the actual response
        	        return fn(*args, **kwargs)

        return _enable_cors


app = bottle.app()

@app.post('/classifier')
def index():

    sentence = request.body.read()
    # print(post_data)
    predict = loaded_model.predict(vectorizer.transform([sentence]))
    
    return str(predict[0])


@app.get('/classifier')
def index():
    return str('classifier')


app.install(EnableCors())
if os.environ.get('APP_LOCATION') == 'heroku':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    app.run(host='localhost', port=8080, debug=True)
