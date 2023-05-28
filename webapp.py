from flask import Flask, render_template, request
import pandas as pd
import pickle

# list of columns accoring to type
cat_col = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Complain', 'Card Type']
num_col = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned']

# calling stored pickle objects
with open('./pickleFiles/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('./pickleFiles/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('./pickleFiles/logitModel.pkl', 'rb') as file:
    model = pickle.load(file)

with open('./pickleFiles/threshold.pkl', 'rb') as file:
    threshold = pickle.load(file)

# pre-processing the input data 
def inputPreProc(df):
    # scaling
    df[num_col] = scaler.transform(df[num_col])

    # encoded matrix
    encoded = encoder.transform(df[cat_col]).toarray()

    # preparing the column names
    all_cat = encoder.categories_
    categories = []
    for i in range(len(cat_col)):
        categories = categories + [f'{cat_col[i]}_{x}' for x in list(all_cat[i])[1:]]
    
    encodedDf = pd.DataFrame(encoded, columns=categories)
    dfnew = df.copy()
    return pd.concat([dfnew.reset_index(drop=True), encodedDf.reset_index(drop=True)], axis=1).drop(cat_col, axis=1)

# prediction of the output 
def predOutput(df):
    df1 = df.copy()
    df.loc[:, "const"] = 1
    df2 = df["const"]

    df = pd.concat([df2, df1], axis=1)
    pred_proba = list(model.predict(df))[0]
    message = "This Customer will leave the bank :( " if pred_proba>threshold else "This Customer will stay with the bank :) "
    conf = pred_proba if pred_proba>threshold else 1-pred_proba
    return(conf, message)

# flask app 
app = Flask(__name__)

# flask fuctionality
@app.route('/', methods=["GET", "POST"])
def home():
    message = "You have not inserted anything yet"
    conf = 0
    try:
        if request.method == "POST":
            inputDf = pd.DataFrame([request.form.to_dict()])
            col_to_float = num_col+['HasCrCard', 'IsActiveMember', 'Complain']
            inputDf[col_to_float] = inputDf[col_to_float].astype('float')
            inputDf = inputPreProc(inputDf)
            conf, message= predOutput(inputDf)
    except KeyError:
        message = "Not all athe inputs are inserted"
    return render_template('webapphome.html', conf=round(100*conf, 2), message=message)

app.run(debug=True)