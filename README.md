# Chunking Strategy

> <i>A try to cut different context into chunks for preparation for RAG retrieval.</i>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-DeepSeek-111827)](https://api-docs.deepseek.com/)
[![Document Parsing](https://img.shields.io/badge/Document%20Parsing-MinerU-6366F1)](https://mineru.net/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 1. Overview

In RAG, the chunking strategy is of great importance. This project is an attempt to chunk context and make the result visible ([the chunking result](./output/chunks_preview.html)). The local result is shown below:

<p align="center">
<img src="./resource/pictures/chunk_insight.png" alt="The local result" width="850">
<br>
<i>Fig.1 The local result</i>
</p>

## 2. Examples

This project processes a PDF file with contexts of pure words, combined with the project running process. The detailed steps are as follows:

**Step 1: Configure `config.json` according to [config_example.json](./config_example.json)**
<br>
In `config_example.json`, the paths for working, Python, and other parameters are assigned. This is important for code running in the following steps.

**Step 2: Run [main.py](./main.py)**

***

**Step 3: Remove empty content**
<br>
Chunking a paper by paragraph is a common method, but it is not valid for abnormal papers, such as soft paragraph chunking. Therefore, this project chunks the paper through punctuation and removes empty content.

**Step 4: Connect different pages and record page numbers**
<br>
To keep the context continuous, different pages need to be connected. Meanwhile, to know the source of the chunks, the page numbers should be recorded.

**Step 5: Multi-level punctuation chunking**
<br>
The first level is punctuation for sentence endings, such as `.?!`; the second level is punctuation for clause endings, such as `:;`; the third level is punctuation for short pauses, such as `、`; and the last level is hard chunking of the context. Meanwhile, overlap is set for the connection between chunks.

**Step 6: Page recording**
<br>
A chunk may come from different pages, and the message for the chunk should include paper ID, chunk ID, pages, the chunk content, and so on.

## 3. License

MIT License.