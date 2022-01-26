import json

from django.http import JsonResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler

from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def getQueries(request):
    path = os.getcwd()
    df = pd.read_pickle(os.path.join(path, 'result/testing_data.pkl'))
    return JsonResponse({
        'queries': list(zip(df['name'].tolist(),  df['author'].tolist()))
    })


@csrf_exempt
def getPrediction(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    query_name = body['query']
    path = os.getcwd()
    path = os.path.join(path, 'result/classifier')
    infile = open(path, 'rb')
    clf = pickle.load(infile)
    path = os.getcwd()
    scaler = StandardScaler()
    df = pd.read_pickle(os.path.join(path, 'result/testing_data.pkl'))
    scaler.fit_transform(df.iloc[:, 2:-1])
    df = pd.read_pickle(os.path.join(path, 'result/testing_data.pkl'))
    df = df.fillna(value=0)
    query = df.loc[df['name'] == query_name]
    true_label = query.iloc[:, -1].tolist()
    book = query.iloc[:, 0].tolist()[0]
    author = query.iloc[:, 1].tolist()[0]
    query = query.iloc[:, 2:-1]
    query = scaler.transform(query)

    pred = clf.predict(query).tolist()[0]
    return JsonResponse({
        'prediction': pred,
        'true': true_label,
        'book': book,
        'author': author
    })


def getResult(request):
    return render(request, 'genreClassification.html', {'fetched': [{
                    "bookName": "The Pupil",
                    "author": "pg10067.epub",
                    "genre": "Detective and Mystery"
                }, {
                    "bookName": "The Heart of the Range",
                    "author": "pg10473.epub",
                    "genre": "Western Stories"
                }, {
                    "bookName": "At Love's Cost",
                    "author": "pg1032.epub",
                    "genre": "Literary"
                }],
                'evalRes': {
                    "accuracy": "85",
                    "precision": "58",
                    "recall": "57",
                    "fmeasure": "170"
                }})