from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

brand = pd.read_csv('datasets/brand.csv')
ingredients = pd.read_csv('datasets/ingredients.csv')
name = pd.read_csv('datasets/name.csv')
price = pd.read_csv('datasets/price.csv')
product = pd.read_csv('datasets/product.csv')
conditions = pd.read_csv('datasets/conditions.csv')

svc = pickle.load(open('models/svc.pkl','rb'))

app = Flask(__name__)


def helper(predicted_condition):
    brand1 = brand[brand['Resulttype'] == predicted_condition]['Brand']
    brand1 = " ".join([u for u in brand1])
    name1 = name[name['Resulttype'] == predicted_condition ]['Name']
    name1 = " ".join([a for a in name1])
    ingredients1 = ingredients[ingredients['Resulttype'] == predicted_condition]['Ingredients']
    ingredients1=" ".join([w for w in ingredients1])
    product1 = product[product['Resulttype'] == predicted_condition]['Product']
    product1 = " ".join([c for c in product1])
    price1 = price[price['Resulttype'] == predicted_condition]['Price']
    price1 = " ".join([f for f in price1])

    return brand1, name1, ingredients1, product1, price1

conditions_dict = {'combination':0,'Dry':1,'Normal':2,'Oily':3,'Sensitive':4}
result_list = {3: 'combinational', 14: 'ideal', 9: 'dizzy', 0: 'ackward', 6: 'dazzling', 11: 'dull', 1: 'balanced', 2: 'bio', 5: 'cranky', 18:'saggy', 7: 'delicate', 10: 'dual', 22: 'uneven', 17: 'unfair', 8: 'distinct', 16: 'optimal', 21: 'tricky', 4: 'common', 19: 'soaky', 20: 'sticky', 13:'even', 23: 'perfect', 15: 'natural', 12: 'erect'}
def get_predicted_value(patient_conditions):
  input_vector = np.zeros(len(conditions_dict))

  for item in patient_conditions:
    input_vector[conditions_dict[item]]=1
  return result_list[svc.predict([input_vector])[0]]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict' , methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        conditions = request.form.get('conditions')
        user_conditions = [s.strip() for s in conditions.split(',')]
        user_conditions = [sym.strip("[]' ") for sym in user_conditions]
        predicted_condition = get_predicted_value(user_conditions)
        brand1, name1, ingredients1, product1, price1 = helper(predicted_condition)

        return render_template('index.html',predicted_condition=predicted_condition,con_brand=brand1,con_name=name1,con_ingredients=ingredients1,con_product=product1,con_price=price1)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/developer')
def developer():
    return render_template('developer.html')
@app.route('/blog')
def blog():
    return render_template('blog.html')



if __name__=="__main__":
    app.run(debug=True)
    