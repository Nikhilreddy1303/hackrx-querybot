import nltk
import ssl
import os

def download_nltk_data():
    """
    Checks for and downloads required NLTK data packages.
    """
    try:
        # This is a workaround for an SSL certificate issue on some systems
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Use more specific paths for reliability
    required = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords'
    }

    for package_id, path in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK '{package_id}' package...")
            nltk.download(package_id, quiet=True)
            print(f"'{package_id}' downloaded.")