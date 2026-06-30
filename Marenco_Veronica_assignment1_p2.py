{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/veronicamarenco/EDC/blob/main/Marenco_Veronica_assignment1_p2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "45sV_cpzlxNQ"
      },
      "source": [
        "# Assignment 1 - Part 2: Text Preprocessing with NLTK\n",
        "\n",
        "**Course:** Natural Language Processing\n",
        "\n",
        "**Total Points:** 10 points (contributes to 50% of Assignment 1)\n",
        "\n",
        "---\n",
        "\n",
        "## Instructions\n",
        "\n",
        "1. Complete all the functions marked with `# YOUR CODE HERE`\n",
        "2. **DO NOT** change the function names or their signatures\n",
        "3. Each function must return the exact type specified\n",
        "4. Test your functions by running the test cells\n",
        "5. When finished:\n",
        "   - Export this notebook as a Python file (.py)\n",
        "   - **Name the file:** `LASTNAME_FIRSTNAME_assignment1_part2.py`\n",
        "   - Example: `DUPONT_Jean_assignment1_part2.py`\n",
        "   - Push to your GitHub repository\n",
        "   - Send the .py file by email to: **yoroba93@gmail.com**\n",
        "\n",
        "---\n",
        "\n",
        "## Assignment Overview\n",
        "\n",
        "In this assignment, you will use NLTK to analyze the Herman Melville novel **Moby Dick**.\n",
        "\n",
        "You will practice:\n",
        "- Tokenization\n",
        "- Frequency analysis\n",
        "- Stop word removal\n",
        "- Stemming and lemmatization\n",
        "- Building a preprocessing pipeline\n",
        "\n",
        "---"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZLSnhWRslxNS"
      },
      "source": [
        "## Setup"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "KHsWvIzGlxNS",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e6d1bb23-cb59-4e48-b538-0a47bcc1df11"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n",
            "Loaded Moby Dick\n",
            "Raw text length: 1220066 characters\n",
            "First 200 characters: [Moby Dick by Herman Melville 1851]\n",
            "\n",
            "\n",
            "ETYMOLOGY.\n",
            "\n",
            "(Supplied by a Late Consumptive Usher to a Grammar School)\n",
            "\n",
            "The pale Usher--threadbare in coat, heart, body, and brain; I see him\n",
            "now.  He was ever du\n"
          ]
        }
      ],
      "source": [
        "import nltk\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import re\n",
        "\n",
        "# Download required NLTK data\n",
        "nltk.download('punkt', quiet=True)\n",
        "nltk.download('punkt_tab', quiet=True)\n",
        "nltk.download('stopwords', quiet=True)\n",
        "nltk.download('wordnet', quiet=True)\n",
        "nltk.download('averaged_perceptron_tagger', quiet=True)\n",
        "nltk.download('averaged_perceptron_tagger_eng', quiet=True)\n",
        "\n",
        "from nltk.tokenize import word_tokenize, sent_tokenize\n",
        "from nltk.stem import PorterStemmer, WordNetLemmatizer\n",
        "from nltk.corpus import stopwords\n",
        "\n",
        "# Mount Google Drive to access files from Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Load the novel\n",
        "# Assuming 'moby.txt' is in the root of 'My Drive'\n",
        "with open('/content/drive/MyDrive/moby.txt', 'r') as f:\n",
        "    moby_raw = f.read()\n",
        "\n",
        "# Create NLTK Text object\n",
        "moby_tokens = nltk.word_tokenize(moby_raw)\n",
        "text1 = nltk.Text(moby_tokens)\n",
        "\n",
        "print(f\"Loaded Moby Dick\")\n",
        "print(f\"Raw text length: {len(moby_raw)} characters\")\n",
        "print(f\"First 200 characters: {moby_raw[:200]}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EBsamJtolxNT"
      },
      "source": [
        "---\n",
        "\n",
        "## Example Functions\n",
        "\n",
        "These examples show you how to work with the text:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "bRBAnz7tlxNT",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "468da9a6-4d84-4b9c-86b7-81ac9a015729"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Total tokens: 255222\n",
            "Unique tokens: 20639\n",
            "Unique tokens after verb lemmatization: 16908\n"
          ]
        }
      ],
      "source": [
        "# Example 1: Count total tokens\n",
        "def example_one():\n",
        "    return len(nltk.word_tokenize(moby_raw))\n",
        "\n",
        "print(f\"Total tokens: {example_one()}\")\n",
        "\n",
        "# Example 2: Count unique tokens\n",
        "def example_two():\n",
        "    return len(set(nltk.word_tokenize(moby_raw)))\n",
        "\n",
        "print(f\"Unique tokens: {example_two()}\")\n",
        "\n",
        "# Example 3: Lemmatize verbs and count unique\n",
        "def example_three():\n",
        "    lemmatizer = WordNetLemmatizer()\n",
        "    lemmatized = [lemmatizer.lemmatize(w, 'v') for w in text1]\n",
        "    return len(set(lemmatized))\n",
        "\n",
        "print(f\"Unique tokens after verb lemmatization: {example_three()}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YyWlsP35lxNU"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 1 (1 point)\n",
        "\n",
        "**What is the lexical diversity of the text?**\n",
        "\n",
        "Lexical diversity = ratio of unique tokens to total number of tokens\n",
        "\n",
        "*This function should return a float.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "L6sblo-dlxNU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9524e9ba-cb85-4161-e348-231e5eccf89a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Lexical diversity: 0.15688817620148093\n"
          ]
        }
      ],
      "source": [
        "def question_one():\n",
        "    \"\"\"\n",
        "    Calculate the lexical diversity of the text.\n",
        "\n",
        "    Returns:\n",
        "        float: Ratio of unique tokens to total tokens\n",
        "    \"\"\"\n",
        "    # Tokenize the text\n",
        "    tokens = moby_raw.split()\n",
        "\n",
        "    # Count unique tokens\n",
        "    unique_tokens = set(tokens)\n",
        "\n",
        "    # Calculate lexical diversity (unique / total)\n",
        "    lexical_diversity = len(unique_tokens) / len(tokens)\n",
        "\n",
        "    return lexical_diversity\n",
        "\n",
        "# Test your function\n",
        "q1_result = question_one()\n",
        "print(f\"Lexical diversity: {q1_result}\")\n",
        "# Expected: approximately 0.08 (8%)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KkC3Ezr1lxNU"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 2 (1 point)\n",
        "\n",
        "**What percentage of tokens is 'whale' or 'Whale'?**\n",
        "\n",
        "*This function should return a float (percentage, e.g., 0.5 for 0.5%).*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "IYjghUZMlxNU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b20cd744-b1e0-48ec-d290-1f14f01d3829"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Percentage of 'whale'/'Whale': None%\n"
          ]
        }
      ],
      "source": [
        "def question_two():\n",
        "    \"\"\"\n",
        "    Calculate the percentage of tokens that are 'whale' or 'Whale'.\n",
        "\n",
        "    Returns:\n",
        "        float: Percentage of whale tokens\n",
        "    \"\"\"\n",
        "    # YOUR CODE HERE\n",
        "\n",
        "    return None\n",
        "\n",
        "# Test your function\n",
        "q2_result = question_two()\n",
        "print(f\"Percentage of 'whale'/'Whale': {q2_result}%\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def question_two():\n",
        "    \"\"\"\n",
        "    Calculate the percentage of tokens that are 'whale' or 'Whale'.\n",
        "\n",
        "    Returns:\n",
        "        float: Percentage of whale tokens\n",
        "    \"\"\"\n",
        "    # Tokenize the text\n",
        "    tokens = moby_raw.split()\n",
        "\n",
        "    # Count total tokens\n",
        "    total_tokens = len(tokens)\n",
        "\n",
        "    # Count whale tokens (case-insensitive)\n",
        "    whale_count = sum(1 for token in tokens if token.lower() == 'whale')\n",
        "\n",
        "    # Calculate percentage\n",
        "    whale_percentage = (whale_count / total_tokens) * 100\n",
        "\n",
        "    return whale_percentage\n",
        "\n",
        "# Test your function\n",
        "q2_result = question_two()\n",
        "print(f\"Percentage of 'whale'/'Whale': {q2_result}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JDR2wVi-wBg2",
        "outputId": "05c0517f-c687-4ee9-d6b6-f089d48c0e71"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Percentage of 'whale'/'Whale': 0.2490213649011932%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gIMSALrOlxNU"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 3 (1 point)\n",
        "\n",
        "**What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?**\n",
        "\n",
        "*This function should return a list of 20 tuples `(token, frequency)`, sorted in descending order of frequency.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "id": "zxJUnPq0lxNU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "763c3d72-345e-4e6a-9b21-da1e1ba91ec4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "20 most frequent tokens:\n",
            "  the: 13604\n",
            "  of: 6475\n",
            "  and: 5881\n",
            "  a: 4472\n",
            "  to: 4439\n",
            "  in: 3824\n",
            "  that: 2680\n",
            "  his: 2415\n",
            "  I: 1724\n",
            "  with: 1645\n",
            "  as: 1590\n",
            "  was: 1564\n",
            "  is: 1560\n",
            "  it: 1506\n",
            "  he: 1492\n",
            "  for: 1357\n",
            "  all: 1293\n",
            "  at: 1211\n",
            "  this: 1131\n",
            "  by: 1093\n"
          ]
        }
      ],
      "source": [
        "from nltk import FreqDist\n",
        "\n",
        "def question_three():\n",
        "    \"\"\"\n",
        "    Find the 20 most frequent tokens and their frequencies.\n",
        "\n",
        "    Returns:\n",
        "        list: List of 20 tuples (token, frequency) sorted by frequency descending\n",
        "    \"\"\"\n",
        "    # Tokenize the text\n",
        "    tokens = moby_raw.split()\n",
        "\n",
        "    # Create a FreqDist object\n",
        "    freq_dist = FreqDist(tokens)\n",
        "\n",
        "    # Get the 20 most common tokens\n",
        "    most_common_20 = freq_dist.most_common(20)\n",
        "\n",
        "    return most_common_20\n",
        "\n",
        "# Test your function\n",
        "q3_result = question_three()\n",
        "print(\"20 most frequent tokens:\")\n",
        "for token, freq in q3_result:\n",
        "    print(f\"  {token}: {freq}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WF7T_emclxNU"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 4 (1 point)\n",
        "\n",
        "**What tokens have a length greater than 5 and a frequency of more than 150?**\n",
        "\n",
        "*This function should return an alphabetically sorted list of tokens.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "0bzVG091lxNU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "35855cd4-c4aa-43bb-8ffa-aa451f3fb468"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 0 tokens:\n",
            "[]\n"
          ]
        }
      ],
      "source": [
        "def question_four():\n",
        "    \"\"\"\n",
        "    Find tokens with length > 5 and frequency > 150.\n",
        "\n",
        "    Returns:\n",
        "        list: Alphabetically sorted list of tokens\n",
        "    \"\"\"\n",
        "    # YOUR CODE HERE\n",
        "\n",
        "    return []\n",
        "\n",
        "# Test your function\n",
        "q4_result = question_four()\n",
        "print(f\"Found {len(q4_result)} tokens:\")\n",
        "print(q4_result)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aq5xWioBlxNU"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 5 (1 point)\n",
        "\n",
        "**Find the longest word in text1 and its length.**\n",
        "\n",
        "*This function should return a tuple `(longest_word, length)`.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "_gGAn3f8lxNV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f17c450e-ac58-45a6-d9c5-6257e30c717c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Longest word: 'matches?--tinder?--gunpowder?--what' with length 35\n"
          ]
        }
      ],
      "source": [
        "def question_five():\n",
        "    \"\"\"\n",
        "    Find the longest word in the text.\n",
        "\n",
        "    Returns:\n",
        "        tuple: (longest_word, length)\n",
        "    \"\"\"\n",
        "    # Tokenize the text\n",
        "    tokens = moby_raw.split()\n",
        "\n",
        "    # Find the longest word\n",
        "    longest_word = max(tokens, key=len)\n",
        "\n",
        "    # Get its length\n",
        "    length = len(longest_word)\n",
        "\n",
        "    return (longest_word, length)\n",
        "\n",
        "# Test your function\n",
        "q5_result = question_five()\n",
        "print(f\"Longest word: '{q5_result[0]}' with length {q5_result[1]}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qgWOMHkFlxNV"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 6 (1 point)\n",
        "\n",
        "**What unique words (only alphabetic tokens) have a frequency of more than 2000?**\n",
        "\n",
        "Use `isalpha()` to check if the token is a word and not punctuation.\n",
        "\n",
        "*This function should return a list of tuples `(frequency, word)` sorted in descending order of frequency.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "n2l_p-calxNV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1722a3eb-5a35-4c15-b5f3-a81b949b3216"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Words with frequency > 2000:\n",
            "  the: 13604\n",
            "  of: 6475\n",
            "  and: 5881\n",
            "  a: 4472\n",
            "  to: 4439\n",
            "  in: 3824\n",
            "  that: 2680\n",
            "  his: 2415\n"
          ]
        }
      ],
      "source": [
        "from nltk import FreqDist\n",
        "\n",
        "def question_six():\n",
        "    \"\"\"\n",
        "    Find words with frequency > 2000.\n",
        "\n",
        "    Returns:\n",
        "        list: List of tuples (frequency, word) sorted by frequency descending\n",
        "    \"\"\"\n",
        "    # Tokenize the text\n",
        "    tokens = moby_raw.split()\n",
        "\n",
        "    # Create a FreqDist object\n",
        "    freq_dist = FreqDist(tokens)\n",
        "\n",
        "    # Filter words with frequency > 2000\n",
        "    high_freq_words = [(freq, word) for word, freq in freq_dist.items() if freq > 2000]\n",
        "\n",
        "    # Sort by frequency descending\n",
        "    high_freq_words.sort(reverse=True)\n",
        "\n",
        "    return high_freq_words\n",
        "\n",
        "# Test your function\n",
        "q6_result = question_six()\n",
        "print(\"Words with frequency > 2000:\")\n",
        "for freq, word in q6_result:\n",
        "    print(f\"  {word}: {freq}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "koX5MYqQlxNV"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 7 (1 point)\n",
        "\n",
        "**What is the average number of tokens per sentence?**\n",
        "\n",
        "*This function should return a float.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "BFKJLbPUlxNV"
      },
      "outputs": [],
      "source": [
        "def question_seven():\n",
        "    \"\"\"\n",
        "    Calculate the average number of tokens per sentence.\n",
        "\n",
        "    Returns:\n",
        "        float: Average tokens per sentence\n",
        "    \"\"\"\n",
        "    # YOUR CODE HERE\n",
        "    # Hint: Use sent_tokenize for sentences, word_tokenize for words\n",
        "\n",
        "    return None\n",
        "\n",
        "# Test your function\n",
        "q7_result = question_seven()\n",
        "print(f\"Average tokens per sentence: {q7_result}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk.tokenize import sent_tokenize, word_tokenize\n",
        "\n",
        "def question_seven():\n",
        "    \"\"\"\n",
        "    Calculate the average number of tokens per sentence.\n",
        "\n",
        "    Returns:\n",
        "        float: Average tokens per sentence\n",
        "    \"\"\"\n",
        "    # Tokenize text into sentences\n",
        "    sentences = sent_tokenize(moby_raw)\n",
        "\n",
        "    # Count total tokens across all sentences\n",
        "    total_tokens = 0\n",
        "    for sentence in sentences:\n",
        "        tokens = word_tokenize(sentence)\n",
        "        total_tokens += len(tokens)\n",
        "\n",
        "    # Calculate average\n",
        "    average_tokens = total_tokens / len(sentences)\n",
        "\n",
        "    return average_tokens\n",
        "\n",
        "# Test your function\n",
        "q7_result = question_seven()\n",
        "print(f\"Average tokens per sentence: {q7_result}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yQ0n4Qn5yMYh",
        "outputId": "cc691c6c-7615-47f7-c55c-5f396f608464"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Average tokens per sentence: 25.90560292326431\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ElWb2JVulxNV"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 8 (1 point)\n",
        "\n",
        "**Remove stop words from the text and return the 10 most common remaining words.**\n",
        "\n",
        "Only consider alphabetic tokens (use `isalpha()`).\n",
        "\n",
        "*This function should return a list of 10 tuples `(word, frequency)` sorted by frequency descending.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "q_nmhExilxNV"
      },
      "outputs": [],
      "source": [
        "def question_eight():\n",
        "    \"\"\"\n",
        "    Find 10 most common words after removing stop words.\n",
        "\n",
        "    Returns:\n",
        "        list: List of 10 tuples (word, frequency) sorted by frequency descending\n",
        "    \"\"\"\n",
        "    # YOUR CODE HERE\n",
        "    # Hint: Use stopwords.words('english')\n",
        "\n",
        "    return []\n",
        "\n",
        "# Test your function\n",
        "q8_result = question_eight()\n",
        "print(\"10 most common words (excluding stop words):\")\n",
        "for word, freq in q8_result:\n",
        "    print(f\"  {word}: {freq}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk import FreqDist\n",
        "from nltk.corpus import stopwords\n",
        "\n",
        "def question_eight():\n",
        "    \"\"\"\n",
        "    Find 10 most common words after removing stop words.\n",
        "\n",
        "    Returns:\n",
        "        list: List of 10 tuples (word, frequency) sorted by frequency descending\n",
        "    \"\"\"\n",
        "    # Get English stop words\n",
        "    stop_words = set(stopwords.words('english'))\n",
        "\n",
        "    # Tokenize the text\n",
        "    tokens = moby_raw.split()\n",
        "\n",
        "    # Filter out stop words\n",
        "    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]\n",
        "\n",
        "    # Create FreqDist and get 10 most common\n",
        "    freq_dist = FreqDist(filtered_tokens)\n",
        "    most_common_10 = freq_dist.most_common(10)\n",
        "\n",
        "    return most_common_10\n",
        "\n",
        "# Test your function\n",
        "q8_result = question_eight()\n",
        "print(\"10 most common words (excluding stop words):\")\n",
        "for word, freq in q8_result:\n",
        "    print(f\"  {word}: {freq}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_xtNs8SNyrmK",
        "outputId": "b77ba673-cab3-41e8-ef12-9706a204521f"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "10 most common words (excluding stop words):\n",
            "  one: 750\n",
            "  like: 544\n",
            "  upon: 531\n",
            "  old: 412\n",
            "  would: 406\n",
            "  whale: 392\n",
            "  great: 282\n",
            "  still: 275\n",
            "  seemed: 273\n",
            "  though: 272\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hCbiGyxrlxNV"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 9 (1 point)\n",
        "\n",
        "**Apply Porter stemming to all words and return the 10 most common stems.**\n",
        "\n",
        "Only consider alphabetic tokens.\n",
        "\n",
        "*This function should return a list of 10 tuples `(stem, frequency)` sorted by frequency descending.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "JZumH-FflxNV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "99c5cca2-1441-4b02-d427-0714c61cb863"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "10 most common stems:\n"
          ]
        }
      ],
      "source": [
        "def question_nine():\n",
        "    \"\"\"\n",
        "    Find 10 most common stems using Porter stemmer.\n",
        "\n",
        "    Returns:\n",
        "        list: List of 10 tuples (stem, frequency) sorted by frequency descending\n",
        "    \"\"\"\n",
        "    # YOUR CODE HERE\n",
        "\n",
        "    return []\n",
        "\n",
        "# Test your function\n",
        "q9_result = question_nine()\n",
        "print(\"10 most common stems:\")\n",
        "for stem, freq in q9_result:\n",
        "    print(f\"  {stem}: {freq}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk import FreqDist\n",
        "from nltk.stem import PorterStemmer\n",
        "\n",
        "def question_nine():\n",
        "    \"\"\"\n",
        "    Find 10 most common stems using Porter stemmer.\n",
        "\n",
        "    Returns:\n",
        "        list: List of 10 tuples (stem, frequency) sorted by frequency descending\n",
        "    \"\"\"\n",
        "    # Initialize Porter stemmer\n",
        "    stemmer = PorterStemmer()\n",
        "\n",
        "    # Tokenize the text\n",
        "    tokens = moby_raw.split()\n",
        "\n",
        "    # Apply stemming to each token\n",
        "    stems = [stemmer.stem(token) for token in tokens]\n",
        "\n",
        "    # Create FreqDist and get 10 most common\n",
        "    freq_dist = FreqDist(stems)\n",
        "    most_common_10 = freq_dist.most_common(10)\n",
        "\n",
        "    return most_common_10\n",
        "\n",
        "# Test your function\n",
        "q9_result = question_nine()\n",
        "print(\"10 most common stems:\")\n",
        "for stem, freq in q9_result:\n",
        "    print(f\"  {stem}: {freq}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xrF8Kxj0y2hh",
        "outputId": "1c4fceb9-828d-48ea-8415-95df13a3b7dc"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "10 most common stems:\n",
            "  the: 14226\n",
            "  of: 6548\n",
            "  and: 6240\n",
            "  a: 4597\n",
            "  to: 4518\n",
            "  in: 4058\n",
            "  that: 2744\n",
            "  hi: 2485\n",
            "  it: 2138\n",
            "  i: 1724\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bPWuTWS4lxNV"
      },
      "source": [
        "---\n",
        "\n",
        "## Question 10 (1 point)\n",
        "\n",
        "**Create a complete preprocessing function that:**\n",
        "1. Tokenizes the text\n",
        "2. Converts to lowercase\n",
        "3. Removes non-alphabetic tokens\n",
        "4. Removes stop words\n",
        "5. Applies lemmatization\n",
        "\n",
        "Apply this function to the first 1000 characters of Moby Dick.\n",
        "\n",
        "*This function should return a list of preprocessed tokens.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OwsSDWFqlxNV"
      },
      "outputs": [],
      "source": [
        "def question_ten():\n",
        "    \"\"\"\n",
        "    Preprocess the first 1000 characters of Moby Dick.\n",
        "\n",
        "    Returns:\n",
        "        list: List of preprocessed tokens\n",
        "    \"\"\"\n",
        "    text = moby_raw[:1000]\n",
        "    stop_words = set(stopwords.words('english'))\n",
        "    lemmatizer = WordNetLemmatizer()\n",
        "\n",
        "    # YOUR CODE HERE\n",
        "    # Steps:\n",
        "    # 1. Tokenize\n",
        "    # 2. Lowercase\n",
        "    # 3. Keep only alphabetic tokens\n",
        "    # 4. Remove stop words\n",
        "    # 5. Lemmatize\n",
        "\n",
        "    return []\n",
        "\n",
        "# Test your function\n",
        "q10_result = question_ten()\n",
        "print(f\"Number of preprocessed tokens: {len(q10_result)}\")\n",
        "print(f\"First 20 tokens: {q10_result[:20]}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def question_ten():\n",
        "    text = moby_raw[:1000]\n",
        "    stop_words = set(stopwords.words('english'))\n",
        "    lemmatizer = WordNetLemmatizer()\n",
        "\n",
        "    tokens = word_tokenize(text.lower())\n",
        "    tokens = [lemmatizer.lemmatize(token)\n",
        "              for token in tokens\n",
        "              if token.isalpha() and token not in stop_words]\n",
        "\n",
        "    return tokens"
      ],
      "metadata": {
        "id": "aVtNL0FBzLlL"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "O41kaVd0lxNW"
      },
      "source": [
        "---\n",
        "\n",
        "## Summary of Functions for Grading\n",
        "\n",
        "Make sure all these functions are properly implemented before exporting:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "id": "I5HH7HTllxNW",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fa6f3d59-54e7-465c-e3de-e2d86a944d21"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Checking functions...\n",
            "✓ question_one: OK\n",
            "✓ question_two: OK\n",
            "✓ question_three: OK\n",
            "✓ question_four: OK\n",
            "✓ question_five: OK\n",
            "✓ question_six: OK\n",
            "✓ question_seven: OK\n",
            "✓ question_eight: OK\n",
            "✓ question_nine: OK\n",
            "✓ question_ten: OK\n",
            "\n",
            "Done! Export this notebook as .py file when all functions pass.\n"
          ]
        }
      ],
      "source": [
        "# Run this cell to verify all functions exist and return correct types\n",
        "print(\"Checking functions...\")\n",
        "\n",
        "try:\n",
        "    r1 = question_one()\n",
        "    assert isinstance(r1, float), \"question_one should return a float\"\n",
        "    print(\"✓ question_one: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_one: {e}\")\n",
        "\n",
        "try:\n",
        "    r2 = question_two()\n",
        "    assert isinstance(r2, float), \"question_two should return a float\"\n",
        "    print(\"✓ question_two: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_two: {e}\")\n",
        "\n",
        "try:\n",
        "    r3 = question_three()\n",
        "    assert isinstance(r3, list) and len(r3) == 20, \"question_three should return a list of 20 tuples\"\n",
        "    print(\"✓ question_three: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_three: {e}\")\n",
        "\n",
        "try:\n",
        "    r4 = question_four()\n",
        "    assert isinstance(r4, list), \"question_four should return a list\"\n",
        "    print(\"✓ question_four: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_four: {e}\")\n",
        "\n",
        "try:\n",
        "    r5 = question_five()\n",
        "    assert isinstance(r5, tuple) and len(r5) == 2, \"question_five should return a tuple of 2 elements\"\n",
        "    print(\"✓ question_five: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_five: {e}\")\n",
        "\n",
        "try:\n",
        "    r6 = question_six()\n",
        "    assert isinstance(r6, list), \"question_six should return a list\"\n",
        "    print(\"✓ question_six: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_six: {e}\")\n",
        "\n",
        "try:\n",
        "    r7 = question_seven()\n",
        "    assert isinstance(r7, float), \"question_seven should return a float\"\n",
        "    print(\"✓ question_seven: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_seven: {e}\")\n",
        "\n",
        "try:\n",
        "    r8 = question_eight()\n",
        "    assert isinstance(r8, list) and len(r8) == 10, \"question_eight should return a list of 10 tuples\"\n",
        "    print(\"✓ question_eight: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_eight: {e}\")\n",
        "\n",
        "try:\n",
        "    r9 = question_nine()\n",
        "    assert isinstance(r9, list) and len(r9) == 10, \"question_nine should return a list of 10 tuples\"\n",
        "    print(\"✓ question_nine: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_nine: {e}\")\n",
        "\n",
        "try:\n",
        "    r10 = question_ten()\n",
        "    assert isinstance(r10, list), \"question_ten should return a list\"\n",
        "    print(\"✓ question_ten: OK\")\n",
        "except Exception as e:\n",
        "    print(f\"✗ question_ten: {e}\")\n",
        "\n",
        "print(\"\\nDone! Export this notebook as .py file when all functions pass.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VwdXCH1RlxNW"
      },
      "source": [
        "---\n",
        "\n",
        "## Submission Checklist\n",
        "\n",
        "- [ ] All 10 functions are implemented\n",
        "- [ ] All functions return the correct type\n",
        "- [ ] Notebook exported as Python file\n",
        "- [ ] File named: `LASTNAME_FIRSTNAME_assignment1_part2.py`\n",
        "- [ ] Pushed to GitHub repository\n",
        "- [ ] Sent to **yoroba93@gmail.com**"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "nlp-course",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.7"
    },
    "colab": {
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}