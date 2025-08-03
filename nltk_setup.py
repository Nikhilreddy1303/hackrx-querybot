import nltk
import ssl

def download_nltk_data():
    """
    Checks for and downloads required NLTK data packages.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    required_packages = ['punkt', 'punkt_tab', 'stopwords'] # Ensure all needed packages are here
    for package in required_packages:
        try:
            # Check if package is already downloaded.
            # For 'punkt' and 'punkt_tab', NLTK expects them under 'tokenizers/'.
            # For 'stopwords', it expects them under 'corpora/'.
            # The find method is smart enough to handle these common paths.
            nltk.data.find(f'{package}') # Simplified find path, NLTK often resolves this
        except LookupError:
            print(f"Downloading NLTK '{package}' package...")
            nltk.download(package, quiet=True)
            print(f"'{package}' downloaded.")

# This block makes the script executable for pre-downloading
if __name__ == "__main__":
    download_nltk_data()