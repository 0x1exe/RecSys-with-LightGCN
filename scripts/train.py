from model_utils import *
from pyspark.sql import SparkSession
import torch
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
from pyspark.sql.functions import col
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1,org.postgresql:postgresql:42.7.5 pyspark-shell'


spark = SparkSession.builder \
    .appName("RecommendationTraining") \
    .config("spark.jars", "postgresql-42.7.5.jar") \
    .getOrCreate()

db_url = "jdbc:postgresql://127.0.0.1:5432/movie_lens"
db_properties = {
    "user": "postgres",
    "password": "psw689",
    "driver": "org.postgresql.Driver"
}

train_df = spark.read.jdbc(
    url=db_url,
    table="(SELECT * FROM train) as train_data",
    properties=db_properties
)

train_df = train_df.toPandas()
n_users = train_df['user_id_idx'].nunique()
n_items = train_df['item_id_idx'].nunique()
latent_dim = 64
n_layers = 3

lightGCN = LightGCN(train_df, n_users, n_items, n_layers, latent_dim)
optimizer = torch.optim.Adam(lightGCN.parameters(), lr = 0.005)

EPOCHS = 30
BATCH_SIZE = 1024
DECAY = 0.0001
K = 10

loss_list_epoch = []
MF_loss_list_epoch = []
reg_loss_list_epoch = []

recall_list = []
precision_list = []
ndcg_list = []
map_list = []

train_time_list = []
eval_time_list = []

for epoch in tqdm(range(EPOCHS)):
    n_batch = int(len(train_df)/BATCH_SIZE)

    final_loss_list = []
    MF_loss_list = []
    reg_loss_list = []

    best_ndcg = -1

    train_start_time = time.time()
    lightGCN.train()
    for batch_idx in range(n_batch):

        optimizer.zero_grad()

        users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items)

        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = lightGCN.forward(users, pos_items, neg_items)

        mf_loss, reg_loss = bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)
        reg_loss = DECAY * reg_loss
        final_loss = mf_loss + reg_loss

        final_loss.backward()
        optimizer.step()

        final_loss_list.append(final_loss.item())
        MF_loss_list.append(mf_loss.item())
        reg_loss_list.append(reg_loss.item())


    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    lightGCN.eval()
    with torch.no_grad():

        final_user_Embed, final_item_Embed, initial_user_Embed,initial_item_Embed = lightGCN.propagate_through_layers()
        test_topK_recall,  test_topK_precision, test_topK_ndcg, test_topK_map  = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train_df, test, K)


    if test_topK_ndcg > best_ndcg:
        best_ndcg = test_topK_ndcg

        torch.save(final_user_Embed, 'final_user_Embed.pt')
        torch.save(final_item_Embed, 'final_item_Embed.pt')
        torch.save(initial_user_Embed, 'initial_user_Embed.pt')
        torch.save(initial_item_Embed, 'initial_item_Embed.pt')


    eval_time = time.time() - train_end_time

    loss_list_epoch.append(round(np.mean(final_loss_list),4))
    MF_loss_list_epoch.append(round(np.mean(MF_loss_list),4))
    reg_loss_list_epoch.append(round(np.mean(reg_loss_list),4))

    recall_list.append(round(test_topK_recall,4))
    precision_list.append(round(test_topK_precision,4))
    ndcg_list.append(round(test_topK_ndcg,4))
    map_list.append(round(test_topK_map,4))

    train_time_list.append(train_time)
    eval_time_list.append(eval_time)

