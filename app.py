from flask import Flask, render_template, url_for, request, redirect
#databases:
from flask_sqlalchemy import SQLAlchemy


#need to set up the applications
app= Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db' #use for the database- relative path(resign in project location) --> results stored in test.db file

#initialize our database with settings from our app
db = SQLAlchemy(app) 

class TestModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable =False)
    def __repr__(self):
        return '<Task %r>' % self.id

#need to create an index route so that we can browse to the URL- requires set up routes:
@app.route('/', methods=['POST', 'GET'])

#define the functions for that route:
def index():
    return render_template("index.html")

class Method(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    content= db.Column(db.String(200), nullable= False)

if __name__ == "__main__":
    app.run(debug=True) #shows errors on page instead of 404 message