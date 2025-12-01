# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.

from typing import Union, Iterable, List
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from kag.interface import VectorizeModelABC, EmbeddingVector
from typing import Callable
import logging

# Default batch size for embedding API calls
DEFAULT_BATCH_SIZE = 10

logger = logging.getLogger(__name__)


@VectorizeModelABC.register("openai")
class OpenAIVectorizeModel(VectorizeModelABC):
    """
    A class that extends the VectorizeModelABC base class.
    It invokes OpenAI or OpenAI-compatible embedding services to convert texts into embedding vectors.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
        base_url: str = "",
        vector_dimensions: int = None,
        timeout: float = None,
        max_rate: float = 1000,
        time_period: float = 1,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs,
    ):
        """
        Initializes the OpenAIVectorizeModel instance.

        Args:
            model (str, optional): The model to use for embedding. Defaults to "text-embedding-3-small".
            api_key (str, optional): The API key for accessing the OpenAI service. Defaults to "".
            base_url (str, optional): The base URL for the OpenAI service. Defaults to "".
            vector_dimensions (int, optional): The number of dimensions for the embedding vectors. Defaults to None.
            batch_size (int, optional): Maximum number of texts to process in a single API call. Defaults to 10.
        """
        api_key = api_key if api_key else "abc123"
        name = self.generate_key(base_url, model, api_key)
        super().__init__(name, vector_dimensions, max_rate, time_period)
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
        self.aclient = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=self.timeout
        )

    @classmethod
    def generate_key(cls, base_url, model, api_key, *args, **kwargs) -> str:
        return f"{cls}_{base_url}_{model}_{api_key}"

    def _split_into_batches(self, items: List, batch_size: int) -> List[List]:
        """Split a list into batches of specified size."""
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def vectorize(
        self, texts: Union[str, Iterable[str]]
    ) -> Union[EmbeddingVector, Iterable[EmbeddingVector]]:
        """
        Vectorizes a text string into an embedding vector or multiple text strings into multiple embedding vectors.

        Args:
            texts (Union[str, Iterable[str]]): The text or texts to vectorize.

        Returns:
            Union[EmbeddingVector, Iterable[EmbeddingVector]]: The embedding vector(s) of the text(s).
        """

        try:
            # Handle empty strings in the input
            if isinstance(texts, list):
                # Create a map of original indices to track empty strings
                empty_indices = {i: text.strip() == "" for i, text in enumerate(texts)}
                # Filter out empty strings for the API call
                filtered_texts = [
                    text for i, text in enumerate(texts) if not empty_indices[i]
                ]

                if not filtered_texts:
                    return [[] for _ in texts]  # Return empty vectors for all inputs

                # Split filtered texts into batches to avoid API batch size limits
                batches = self._split_into_batches(filtered_texts, self.batch_size)
                embeddings = []
                for batch in batches:
                    results = self.client.embeddings.create(
                        input=batch, model=self.model
                    )
                    embeddings.extend([item.embedding for item in results.data])

                # Reconstruct the results with empty lists for empty strings
                full_results = []
                embedding_idx = 0

                for i in range(len(texts)):
                    if empty_indices[i]:
                        full_results.append([])  # Empty embedding for empty string
                    else:
                        full_results.append(embeddings[embedding_idx])
                        embedding_idx += 1

                return full_results
            elif isinstance(texts, str) and not texts.strip():
                return []  # Return empty vector for empty string
            else:
                results = self.client.embeddings.create(input=texts, model=self.model)
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"input: {texts}")
            logger.error(f"model: {self.model}")
            logger.error(f"timeout: {self.timeout}")
            return None
        results = [item.embedding for item in results.data]
        if isinstance(texts, str):
            assert len(results) == 1
            return results[0]
        else:
            assert len(results) == len(texts)
            return results

    async def avectorize(
        self, texts: Union[str, Iterable[str]]
    ) -> Union[EmbeddingVector, Iterable[EmbeddingVector]]:
        """
        Vectorizes a text string into an embedding vector or multiple text strings into multiple embedding vectors.

        Args:
            texts (Union[str, Iterable[str]]): The text or texts to vectorize.

        Returns:
            Union[EmbeddingVector, Iterable[EmbeddingVector]]: The embedding vector(s) of the text(s).
        """
        async with self.limiter:
            try:
                if isinstance(texts, list):
                    # Handle empty strings by replacing them with placeholder
                    processed_texts = [text if text.strip() != "" else "none" for text in texts]

                    # Split into batches to avoid API batch size limits
                    batches = self._split_into_batches(processed_texts, self.batch_size)
                    all_embeddings = []
                    for batch in batches:
                        results = await self.aclient.embeddings.create(
                            input=batch, model=self.model
                        )
                        all_embeddings.extend([item.embedding for item in results.data])

                    assert len(all_embeddings) == len(texts)
                    return all_embeddings
                else:
                    # Handle single string input
                    if isinstance(texts, str) and not texts.strip():
                        texts = "none"
                    results = await self.aclient.embeddings.create(
                        input=texts, model=self.model
                    )
                    results = [item.embedding for item in results.data]
                    assert len(results) == 1
                    return results[0]
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.error(f"input: {texts}")
                logger.error(f"model: {self.model}")
                logger.error(f"timeout: {self.timeout}")
                return None


@VectorizeModelABC.register("azure_openai")
class AzureOpenAIVectorizeModel(VectorizeModelABC):
    """A class that extends the VectorizeModelABC base class.
    It invokes Azure OpenAI or Azure OpenAI-compatible embedding services to convert texts into embedding vectors.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-ada-002",
        api_version: str = "2024-12-01-preview",
        vector_dimensions: int = None,
        timeout: float = None,
        azure_deployment: str = None,
        azure_ad_token: str = None,
        azure_ad_token_provider: Callable = None,
        max_rate: float = 1000,
        time_period: float = 1,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initializes the AzureOpenAIVectorizeModel instance.

        Args:
            model (str, optional): The model to use for embedding. Defaults to "text-embedding-3-small".
            api_key (str, optional): The API key for accessing the Azure OpenAI service. Defaults to "".
            api_version (str): The API version for the Azure OpenAI API (eg. "2024-12-01-preview, 2024-10-01-preview,2024-05-01-preview").
            base_url (str, optional): The base URL for the Azure OpenAI service. Defaults to "".
            vector_dimensions (int, optional): The number of dimensions for the embedding vectors. Defaults to None.
            azure_ad_token: Your Azure Active Directory token, https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id
            azure_ad_token_provider: A function that returns an Azure Active Directory token, will be invoked on every request.
            azure_deployment: A model deployment, if given sets the base client URL to include `/deployments/{azure_deployment}`.
                Note: this means you won't be able to use non-deployment endpoints. Not supported with Assistants APIs.
            batch_size (int, optional): Maximum number of texts to process in a single API call. Defaults to 10.
        """
        name = self.generate_key(api_key, base_url, model)
        super().__init__(name, vector_dimensions, max_rate, time_period)
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.client = AzureOpenAI(
            api_key=api_key,
            base_url=base_url,
            azure_deployment=azure_deployment,
            model=model,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            timeout=self.timeout,
        )
        self.aclient = AsyncAzureOpenAI(
            api_key=api_key,
            base_url=base_url,
            azure_deployment=azure_deployment,
            model=model,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            timeout=self.timeout,
        )

    @classmethod
    def generate_key(cls, base_url, api_key, model, *args, **kwargs) -> str:
        return f"{cls}_{base_url}_{api_key}_{model}"

    def _split_into_batches(self, items: List, batch_size: int) -> List[List]:
        """Split a list into batches of specified size."""
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def vectorize(
        self, texts: Union[str, Iterable[str]]
    ) -> Union[EmbeddingVector, Iterable[EmbeddingVector]]:
        """
        Vectorizes a text string into an embedding vector or multiple text strings into multiple embedding vectors.

        Args:
            texts (Union[str, Iterable[str]]): The text or texts to vectorize.

        Returns:
            Union[EmbeddingVector, Iterable[EmbeddingVector]]: The embedding vector(s) of the text(s).
        """
        if isinstance(texts, str):
            results = self.client.embeddings.create(input=texts, model=self.model)
            results = [item.embedding for item in results.data]
            assert len(results) == 1
            return results[0]
        else:
            # Split texts into batches to avoid API batch size limits
            texts_list = list(texts)
            batches = self._split_into_batches(texts_list, self.batch_size)
            all_embeddings = []
            for batch in batches:
                results = self.client.embeddings.create(input=batch, model=self.model)
                all_embeddings.extend([item.embedding for item in results.data])
            assert len(all_embeddings) == len(texts_list)
            return all_embeddings

    async def avectorize(
        self, texts: Union[str, Iterable[str]]
    ) -> Union[EmbeddingVector, Iterable[EmbeddingVector]]:
        """
        Vectorizes a text string into an embedding vector or multiple text strings into multiple embedding vectors.

        Args:
            texts (Union[str, Iterable[str]]): The text or texts to vectorize.

        Returns:
            Union[EmbeddingVector, Iterable[EmbeddingVector]]: The embedding vector(s) of the text(s).
        """
        async with self.limiter:
            if isinstance(texts, str):
                results = await self.aclient.embeddings.create(
                    input=texts, model=self.model
                )
                results = [item.embedding for item in results.data]
                assert len(results) == 1
                return results[0]
            else:
                # Split texts into batches to avoid API batch size limits
                texts_list = list(texts)
                batches = self._split_into_batches(texts_list, self.batch_size)
                all_embeddings = []
                for batch in batches:
                    results = await self.aclient.embeddings.create(
                        input=batch, model=self.model
                    )
                    all_embeddings.extend([item.embedding for item in results.data])
                assert len(all_embeddings) == len(texts_list)
                return all_embeddings


if __name__ == "__main__":
    vectorize_model = OpenAIVectorizeModel(
        model="bge-m3", base_url="http://localhost:11434/v1"
    )
    texts = ["Hello, world!", "Hello, world!", "Hello, world!"]
    embeddings = vectorize_model.vectorize("texts")
    print(embeddings)
