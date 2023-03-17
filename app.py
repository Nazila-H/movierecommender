from flask import Flask,render_template,request
from recommender import recommend_random,recommend_with_NMF,recommend_neighborhood
from utils import movies
from sklearn.decomposition import NMF 
app = Flask(__name__)

@app.route('/')
def hello():
    # print(movies.title.to_list())
    return render_template('index.html', name='Arjun', movies=movies.title.to_list())

@app.route('/movies')
def recommendation():
    print(request.args)

    if request.args.get('option') =='Random':
        recommendation_list = recommend_random()
        print(recommendation_list)


        return render_template('recommend.html', recommendation=recommendation_list)
    
    if request.args.get('option')=='NMF':
        titles = request.args.getlist('title')
        ratings = request.args.getlist('rating')

        query = dict(zip(titles,ratings))
        print(query)

        for movie in query:
            #print(query[movie])
            #print(float(query[movie]))
            query[movie] = float(query[movie])
        
        print('-----',query)
        recommendation_list = recommend_with_NMF(query)
        return render_template('recommend.html', recommendation=recommendation_list)
    
    else:
        recommendation_list = recommend_neighborhood()
        return f'Function not defined'

if __name__=='__main__':
    app.run(port=5000,debug=True)
    
