#Inicio importando as bibliotecas necessárias e
#destaco a tensorflow de análise de aprendizado de máquina fazendo a relação das camadas que necessito e coloco o modelo necessário
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import Input, Embedding, Flatten, StringLookup, Normalization, Concatenate, Dense, Dot
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

#Faço a leitura da base de dados e observo os atributos que ele tem
dataset_path = Path(__file__).resolve().parent.parent / 'content' / 'Dataset_avaliacao.csv'
df = pd.read_csv(dataset_path)
df

df_treino, df_teste = train_test_split(df, test_size=0.2, random_state=42)

#Para a torre do usuário determino as User_ID, User_Age e User_State
#E assim detemino seu modo de importação
in_user_id = Input(shape=(1,), name="User_ID", dtype=tf.string)
in_user_age = Input(shape=(1,), name="User_Age", dtype=tf.float32)
in_user_state = Input(shape=(1,), name="User_State", dtype=tf.string)

#Para a torre do item determino as Book_ID, Book_title, Book_Genre, Book_Pages e Book_Price
#E assim detemino seu modo de importação
in_book_id = Input(shape=(1,), name="Book_ID", dtype=tf.string)
in_book_title = Input(shape=(1,), name="Book_Title", dtype=tf.string)
in_book_genre = Input(shape=(1,), name="Book_Genre", dtype=tf.string)
in_book_pages = Input(shape=(1,), name="Book_Pages", dtype=tf.float32)
in_book_price = Input(shape=(1,), name="Book_Price", dtype=tf.float32)

#Após preciso que redes neurais transformem a palavras em vetores numéricos com função embedding
def criar_embedding(entrada, vocabulario, dim_emb, nome):
    """Mapeia string para ID e depois para Vetor (cria o Balde Zero para OOV)"""
    lookup = StringLookup(vocabulary=vocabulario, mask_token=None, name=f"Lookup_{nome}")(entrada)
    emb = Embedding(input_dim=len(vocabulario) + 1, output_dim=dim_emb, name=f"Emb_{nome}")(lookup)
    return Flatten()(emb)

#Crio embedding da torre do usuário
emb_user = criar_embedding(in_user_id, df_treino['User_ID'].unique(), 8, "User")
emb_cat = criar_embedding(in_user_state, df_treino['User_State'].unique(), 8, "State")

#Crio embedding da torre do item
emb_book = criar_embedding(in_book_id, df_treino['Book_ID'].unique(), 8, "Book")
emb_title = criar_embedding(in_book_title, df_treino['Book_Title'].unique(), 8, "Title")
emb_genre = criar_embedding(in_book_genre, df_treino['Book_Genre'].unique(), 4, "Genre")

#Agora faço a normalização para que fique em escala semelhante ajudando, também, a contribuição de cada característica
def normalizar(entrada, dados_coluna, nome):
    norm = Normalization(name=f"Norm_{nome}")
    norm.adapt(dados_coluna.values)
    return norm(entrada)

norm_age = normalizar(in_user_age, df_treino[['User_Age']], "Age")
norm_pages = normalizar(in_book_pages, df_treino[['Book_Pages']], "Pages")
norm_price = normalizar(in_book_price, df_treino[['Book_Price']], "Price")

#Criando a arquitetura das Duas torres criando os "super vetores"

# Torre do Usuário: Vou concatentar os valores de ID, Estado e Idade[4]
user_features = Concatenate()([emb_user, emb_cat, norm_age])
torre_user = Dense(32, activation='relu')(user_features)
vetor_final_user = Dense(16, activation='linear')(torre_user)

# Torre do Item: co ID do Livro, Título, Gênero, Páginas e Preço [4]
item_features = Concatenate()([emb_book, emb_title, emb_genre, norm_pages, norm_price])
torre_item = Dense(32, activation='relu')(item_features)
vetor_final_item = Dense(16, activation='linear')(torre_item)

# 6. Similaridade e Predição Final
# Cálculo da similaridade geométrica via Produto Escalar [3]
score_similaridade = Dot(axes=1)([vetor_final_user, vetor_final_item])

# Saída Sigmoid para prever a probabilidade do "Liked" [5]
probabilidade = Dense(1, activation='sigmoid', name="Probabilidade_Like")(score_similaridade)

# Produto Escalar
modelo = Model(
    inputs=[in_user_id, in_user_age, in_user_state, in_book_id, in_book_title, in_book_genre, in_book_pages, in_book_price],
    outputs=probabilidade
)
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo.summary()

dict_treino = {
    "User_ID": df_treino['User_ID'].values,
    "User_Age": df_treino['User_Age'].values,
    "User_State": df_treino['User_State'].values,
    "Book_ID": df_treino['Book_ID'].values,
    "Book_Title": df_treino['Book_Title'].values,
    "Book_Genre": df_treino['Book_Genre'].values,
    "Book_Pages": df_treino['Book_Pages'].values,
    "Book_Price": df_treino['Book_Price'].values
}

modelo.fit(x=dict_treino, y=df_treino['Liked'].values, epochs=50, batch_size=4, verbose=1)

#Predição com Dados de Teste
dict_teste = {col: df_teste[col].values for col in dict_treino.keys()}
predicoes = modelo.predict(dict_teste)

print("\nExemplo de Predições (Probabilidade de Like):")
print(predicoes[:5])