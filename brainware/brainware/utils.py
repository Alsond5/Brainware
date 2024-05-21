from django.core.cache import cache
from django.conf import settings
import fasttext

from transformers import *
import torch

from sklearn.metrics.pairwise import cosine_similarity

import numpy
import json
import math

fasttext_cache_name = "fasttext_ids"
scibert_cache_name = "scibert_ids"

fasttext_model = fasttext.load_model(settings.FASTTEXT_MODEL)

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

def get_articles_from_articles_cache(cache_name: str, dataset: dict, page: int):
    articles = cache.get(cache_name)
    
    if articles is None:
        return None
    
    responses = articles[page * 5:(page * 5) + 5]

    return [{
        "id": int(dataset[article["index"]]["id"]),
        "index": article["index"],
        "title": dataset[article["index"]]["title"],
        "abstract": dataset[article["index"]]["abstract"],
        "similarityRate": article["similarityRate"],
        "keys": dataset[article["index"]]["keys"],
        "is_recommended": True if page == 0 else False
    } for article in responses]

def get_articles_from_fasttext_model(interests: str, article_ids: list[int], dataset: dict, page: int):
    model = fasttext_model
    seperator = ","
    
    parts = interests.split(seperator)

    user_vectors = [model.get_word_vector(interest.strip()) for interest in parts]
    user_vectors.extend([dataset[index]["fasttext_vector"] for index in article_ids])
    user_avarage_vector = sum(user_vectors) / len(user_vectors)

    print(user_avarage_vector)

    vectors = [numpy.array(article["fasttext_vector"]) for article in dataset]

    fasttext_similarities = cosine_similarity([user_avarage_vector], vectors)
    fasttext_results = fasttext_similarities.argsort()[0]

    fasttext_articles = [{
        "id": int(dataset[index]["id"]),
        "index": index,
        "title": dataset[index]["title"],
        "abstract": dataset[index]["abstract"],
        "similarityRate": fasttext_similarities[0][index] * 100,
        "keys": dataset[index]["keys"],
        "is_recommended": True if page == 0 else False
    } for index in fasttext_results[::-1][page * 5:(page * 5) + 5]]

    cache_data = [{
        "index": index,
        "similarityRate": fasttext_similarities[0][index] * 100
    } for index in fasttext_results[::-1]]

    return fasttext_articles, cache_data

def get_articles_from_scibert_model(interests: str, article_ids: list[int], dataset: dict, page: int):
    model = scibert_model
    tokens = tokenizer(interests, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens)

        if len(article_ids) > 0:
            external_vector = [dataset[index]["scibert_vector"] for index in article_ids]
            external_vector_tensor = torch.tensor(external_vector)

            external_vector_tensor_expanded = external_vector_tensor.unsqueeze(0)

            combined_vectors = torch.cat((outputs.last_hidden_state, external_vector_tensor_expanded), dim=1)

    if len(article_ids) > 0:
        user_avarage_vector = torch.mean(combined_vectors, dim=1).squeeze().numpy().tolist()
    else:
        user_avarage_vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy().tolist()

    vectors = [numpy.array(article["scibert_vector"]) for article in dataset]

    scibert_similarities = cosine_similarity([user_avarage_vector], vectors)
    scibert_results = scibert_similarities.argsort()[0]

    scibert_articles = [{
        "id": int(dataset[index]["id"]),
        "index": index,
        "title": dataset[index]["title"],
        "abstract": dataset[index]["abstract"],
        "similarityRate": scibert_similarities[0][index] * 100,
        "keys": dataset[index]["keys"],
        "is_recommended": True if page == 0 else False
    } for index in scibert_results[::-1][page * 5:(page * 5) + 5]]

    cache_data = [{
        "index": index,
        "similarityRate": scibert_similarities[0][index] * 100
    } for index in scibert_results[::-1]]

    return scibert_articles, cache_data

def get_articles_from_all_models(interests: str, article_ids: list[int], dataset: dict, pages: tuple[int, int]):
    model = fasttext_model
    seperator = ","
    
    parts = interests.split(seperator)

    user_vectors = [model.get_word_vector(interest.strip()) for interest in parts]
    user_vectors.extend([dataset[index]["fasttext_vector"] for index in article_ids])
    user_avarage_vector = sum(user_vectors) / len(user_vectors)

    print(user_avarage_vector)

    vectors = [{
        "fasttext_vectors": numpy.array(article["fasttext_vector"]),
        "scibert_vectors": numpy.array(article["scibert_vector"])
    } for article in dataset]

    fasttext_similarities = cosine_similarity([user_avarage_vector], [vector["fasttext_vectors"] for vector in vectors])
    fasttext_results = fasttext_similarities.argsort()[0]

    tokens = tokenizer(interests, return_tensors="pt")

    with torch.no_grad():
        outputs = scibert_model(**tokens)

        if len(article_ids) > 0:
            external_vector = [dataset[index]["scibert_vector"] for index in article_ids]
            external_vector_tensor = torch.tensor(external_vector)

            external_vector_tensor_expanded = external_vector_tensor.unsqueeze(0)

            combined_vectors = torch.cat((outputs.last_hidden_state, external_vector_tensor_expanded), dim=1)

    if len(article_ids) > 0:
        user_avarage_vector = torch.mean(combined_vectors, dim=1).squeeze().numpy().tolist()
    else:
        user_avarage_vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy().tolist()

    scibert_similarities = cosine_similarity([user_avarage_vector], [vector["scibert_vectors"] for vector in vectors])
    scibert_results = scibert_similarities.argsort()[0]

    fasttext_articles = [{
        "id": int(dataset[index]["id"]),
        "index": index,
        "title": dataset[index]["title"],
        "abstract": dataset[index]["abstract"],
        "similarityRate": fasttext_similarities[0][index] * 100,
        "keys": dataset[index]["keys"],
        "is_recommended": True if pages[0] == 0 else False
    } for index in fasttext_results[::-1][pages[0] * 5:(pages[0] * 5) + 5]]

    scibert_articles = [{
        "id": int(dataset[index]["id"]),
        "index": index,
        "title": dataset[index]["title"],
        "abstract": dataset[index]["abstract"],
        "similarityRate": scibert_similarities[0][index] * 100,
        "keys": dataset[index]["keys"],
        "is_recommended": True if pages[1] == 0 else False
    } for index in scibert_results[::-1][pages[1] * 5:(pages[1] * 5) + 5]]

    fasttext_cache_data = [{
        "index": index,
        "similarityRate": fasttext_similarities[0][index] * 100
    } for index in fasttext_results[::-1]]
    
    scibert_cache_data = [{
        "index": index,
        "similarityRate": scibert_similarities[0][index] * 100
    } for index in scibert_results[::-1]]

    return fasttext_articles, scibert_articles, (fasttext_cache_data, scibert_cache_data)

def get_recommended_articles(interests: str, article_ids: list[int], pages: tuple[int, int]):
    with open(settings.ARTICLES, "r") as file:
        dataset = json.load(file)

    if pages[0] == 0 and pages[1] == 0:
        fasttext_articles, scibert_articles, cache_data = get_articles_from_all_models(interests, article_ids, dataset, pages)
        
        cache.set(fasttext_cache_name, cache_data[0])
        cache.set(scibert_cache_name, cache_data[1])

        total_pages = math.ceil(len(dataset) / 5)
        
        return fasttext_articles, scibert_articles, total_pages
    
    if pages[0] == 0:
        fasttext_articles, cache_data = get_articles_from_fasttext_model(interests, article_ids, dataset, pages[0])

        cache.set(fasttext_cache_name, cache_data)
    else:
        fasttext_articles = get_articles_from_articles_cache(fasttext_cache_name, dataset, pages[0])

        if fasttext_articles is None:
            fasttext_articles, cache_data = get_articles_from_fasttext_model(interests, article_ids, dataset, pages[0])

            cache.set(fasttext_cache_name, cache_data)

    if pages[1] == 0:
        scibert_articles, cache_data = get_articles_from_scibert_model(interests, article_ids, dataset, pages[1])
        
        cache.set(scibert_cache_name, cache_data)
    else:
        scibert_articles = get_articles_from_articles_cache(scibert_cache_name, dataset, pages[1])

        if scibert_articles is None:
            scibert_articles, cache_data = get_articles_from_scibert_model(interests, article_ids, dataset, pages[1])
            
            cache.set(scibert_cache_name, cache_data)

    total_pages = math.ceil(len(dataset) / 5)

    return fasttext_articles, scibert_articles, total_pages