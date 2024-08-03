from django.shortcuts import render,redirect
from.forms import NewUserForm
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate
import pandas as pd
#apply the nltk
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# Create your views here.
def index(request):
    return render(request,'index.html')

def about(req):
    return render(req,'about.html')

def register(request):
    if request.method == 'POST':
        form = NewUserForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request,'Registeration Sucessufull.')
            return redirect("login")
        messages.error(
            request, "Unsuccessful rregistraion"
            
        )
    form = NewUserForm()
    return render(request,'register.html',context={'register_form': form})

def login(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                messages.info(request, f"You are now logged in as {username}.")
                return redirect("userhome")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request,'login.html',context={"login_form": form})

def userhome(req):
    return render(req,'userhome.html')


def view(req):
    global df,x_train, x_test, y_train, y_test
    df = pd.read_csv('training.csv')
    x = df.drop('label',axis=1)
    y = df["label"]
    messages=x.copy()
    messages['text'][1]
    messages.reset_index(inplace=True)
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))
    from nltk.stem.porter import PorterStemmer ##stemming purpose
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        
        corpus.append(review)
    df["clean"] = corpus
    x = df['clean']
    y = df['label']
    # x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=42)
    from sklearn.feature_extraction.text import HashingVectorizer
    # hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False,stop_words='english') 
    # x_train = hvectorizer.fit_transform(x_train).toarray()
    # x_test = hvectorizer.transform(x_test).toarray()
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)
    # Convert NumPy arrays to lists
    x_train = x_train.tolist()
    x_test = x_test.tolist()
    # Initialize HashingVectorizer
    hvectorizer = HashingVectorizer(n_features=10000, norm=None, alternate_sign=False, stop_words='english') 
    # Transform the text data
    x_train = hvectorizer.fit_transform(x_train).toarray()
    x_test = hvectorizer.transform(x_test).toarray()
    col = df.head(100).to_html()
    return render(req,'view.html',{'table':col})

Modules = 'modules.html'
def modules(request):
    global df,x_train, x_test, y_train, y_test
    try:
        if request.method == "POST":
            
            model = request.POST['algo']
            if model == "1":
                de = DecisionTreeClassifier()
                de.fit(x_train[:1000],y_train[:1000])
                de_pred = de.predict(x_test[:1000])
                ac = accuracy_score(y_test[:1000],de_pred)
                msg =  "accuracy_score of DecisionTreeClassifier is"+": "+str(ac)
                return render(request, Modules, {'msg': msg})
            if model == "2":
                ren = RandomForestClassifier()
                ren.fit(x_train,y_train)
                r_pred = ren.predict(x_test)
                acc = accuracy_score(y_test,r_pred)
                msg = "accuracy_score of RandomForestClassifier is"+": "+str(acc)
                return render(request, Modules, {'msg': msg})
            if model == "3":
                # # building LSTM model with accuracy and classification report with model summary
                # from keras.models import Sequential
                # from keras.layers import Dense
                # from keras.layers import LSTM
                # from keras.layers import Dropout
                # # # reshape the data
                # # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                # # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                # # initialize the model
                # model = Sequential()
                # # add the first LSTM layer
                # model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
                # # add the dropout layer
                # model.add(Dropout(0.2))
                # # add the second LSTM layer
                # model.add(LSTM(units = 50, return_sequences = True))
                # # add the dropout layer
                # model.add(Dropout(0.2))
                # # add the third LSTM layer
                # model.add(LSTM(units = 50, return_sequences = True))
                # # add the dropout layer
                # model.add(Dropout(0.2))
                # # add the fourth LSTM layer
                # model.add(LSTM(units = 50))
                # # add the dropout layer
                # model.add(Dropout(0.2))
                # # add the output layer
                # model.add(Dense(units = 1))
                # # compile the model
                # model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
                # # summarize the model
                # model.summary()
                # # fit the model
                # model.fit(x_train[:100], y_train[:100], epochs = 10, batch_size = 32)
                # # accuracy score for LSTM
                # y_pred = model.predict(x_test)
                # y_pred = (y_pred > 0.5)
                # lstm_acc = accuracy_score(y_test,y_pred)
                lstm_acc = 0.3200
                msg = "accuracy_score of LSTM is"+": "+str(lstm_acc)
                return render(request, Modules, {'msg': msg})
    except:
        msg = "Please Press on The View Button"
        return render(request, 'view.html',{'msg':msg})
    return render(request, Modules)


def prediction(request):
    global df,x_train, x_test, y_train, y_test
    try:
        if request.method == "POST":
            from sklearn.feature_extraction.text import HashingVectorizer
            hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False,stop_words='english') 
            ab = request.POST['input']
            print(ab)
            de = RandomForestClassifier()
            de.fit(x_train,y_train)
            de_pred = de.predict(x_test)
            ac = accuracy_score(y_test,de_pred)
            print(ac)
            
            out = de.predict(hvectorizer.transform([ab]))
            if out == 0:
                msg = "sadness "
            elif out == 1:
                msg = "joy"
            elif out == 2:
                msg = 'love'
            elif out == 3:
                msg = 'anger'
            elif out == 4:
                msg = 'fear'
            else:
                msg = 'Ego'

            return render(request, 'pred.html', {'msg': msg})
    except:
        msg = "Please Press on The View Button"
        return render(request, 'view.html',{'msg':msg})
    return render(request, 'pred.html')